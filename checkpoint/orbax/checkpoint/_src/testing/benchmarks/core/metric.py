# Copyright 2026 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metric classes for benchmarking."""

import collections
from collections.abc import MutableMapping
import contextlib
import dataclasses
import json
import linecache  # To show the source code line
import os
import threading
import time
import tracemalloc
from typing import Any

from absl import logging
from clu import metric_writers
from etils import epath
import numpy as np
from orbax.checkpoint._src.testing.benchmarks.core import multihost
from orbax.checkpoint._src.testing.benchmarks.core import run_manifest as run_manifest_lib
import psutil
import tensorstore as ts


class BaseMetric:
  """Base class for a metric type.

  Subclass override knobs:
    OMIT_REGISTRY_KEY_PREFIX: when True, the metric's result keys are taken
      as final and the METRIC_REGISTRY key (e.g. "jax_monitoring") is NOT
      spliced into the TB tag. Use when the metric already namespaces its
      own keys (e.g. JaxMonitoringMetric returns "2_save_breakdown/...").
  """

  OMIT_REGISTRY_KEY_PREFIX: bool = False

  def __init__(self, name: str):
    self.name = name
    self._start_time = 0

  def start(self):
    """Start the metric collection."""
    self._start_time = time.perf_counter()
    logging.info(
        "[process_id=%s] Starting metric: '%s'...",
        multihost.get_process_index(),
        self.name,
    )

  def stop(self) -> dict[str, tuple[Any, str]]:
    """Stop the metric collection and return results."""
    duration = time.perf_counter() - self._start_time
    logging.info(
        "[process_id=%s] Finished metric: '%s' (took %.4fs)",
        multihost.get_process_index(),
        self.name,
        duration,
    )
    return {}


class TimeMetric(BaseMetric):
  """Measures execution time."""

  OMIT_REGISTRY_KEY_PREFIX = True

  def stop(self) -> dict[str, tuple[Any, str]]:
    duration = time.perf_counter() - self._start_time
    results = super().stop()
    results["0_basics/time_s"] = (duration, "s")
    return results


class RssMetric(BaseMetric):
  """Measures RSS memory difference."""

  OMIT_REGISTRY_KEY_PREFIX = True
  _start_rss: float = 0

  def start(self):
    super().start()
    self._start_rss = self._get_process_memory()

  def stop(self) -> dict[str, tuple[Any, str]]:
    rss_diff = self._get_process_memory() - self._start_rss
    results = super().stop()
    results["0_basics/host_rss_diff_mb"] = (rss_diff, "MB")
    return results

  def _get_process_memory(self):
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


class TracemallocMetric(BaseMetric):
  """Measures memory allocation differences using tracemalloc."""

  OMIT_REGISTRY_KEY_PREFIX = True
  _lock = threading.Lock()
  _active_count = 0
  _start_snapshot: Any = None
  _start_peak: int = 0

  def start(self):
    super().start()
    with TracemallocMetric._lock:
      if TracemallocMetric._active_count == 0:
        tracemalloc.start()
      TracemallocMetric._active_count += 1
    self._start_snapshot = tracemalloc.take_snapshot()
    _, self._start_peak = tracemalloc.get_traced_memory()

  def stop(self) -> dict[str, tuple[Any, str]]:
    results = super().stop()
    if self._start_snapshot is None:
      return results

    _, end_peak = tracemalloc.get_traced_memory()
    end_snapshot = tracemalloc.take_snapshot()

    with TracemallocMetric._lock:
      TracemallocMetric._active_count -= 1
      if TracemallocMetric._active_count == 0:
        tracemalloc.stop()

    peak_diff = end_peak - self._start_peak
    results["7_memory/tracemalloc_peak_diff_mb"] = (peak_diff / (1024**2), "MB")

    self._log_tracemalloc_snapshot_diff(
        self.name,
        multihost.get_process_index(),
        self._start_snapshot,
        end_snapshot,
        top_n=15,
        peak=peak_diff,
    )
    return results

  def _log_tracemalloc_snapshot_diff(
      self,
      name: str,
      process_index: int,
      snapshot1: tracemalloc.Snapshot,
      snapshot2: tracemalloc.Snapshot,
      top_n: int,
      peak: float,
  ):
    """Compares two tracemalloc snapshots and logs the differences using logging.info.

    Args:
        name: The name of the metric.
        process_index: The process index of the metric.
        snapshot1: The earlier tracemalloc Snapshot.
        snapshot2: The later tracemalloc Snapshot.
        top_n: Number of top differences to log, sorted by memory difference.
        peak: The peak memory usage of the process.
    """
    if not isinstance(snapshot1, tracemalloc.Snapshot) or not isinstance(
        snapshot2, tracemalloc.Snapshot
    ):
      logging.error(
          "Invalid input: Both inputs must be tracemalloc.Snapshot objects."
      )
      return

    logging.info("--- Comparing tracemalloc snapshots for %s ---", name)
    stats = snapshot2.compare_to(snapshot1, "lineno")

    if not stats:
      logging.info(
          "[process_id=%s][name=%s] No memory differences found between"
          " snapshots.",
          process_index,
          name,
      )
      return

    total_diff_bytes = sum(stat.size_diff for stat in stats)
    total_new_allocs = sum(stat.count_diff for stat in stats)

    logging.info(
        "[process_id=%s][name=%s] Total memory difference: %.2f KiB, peak:"
        " %.4f GiB",
        process_index,
        name,
        total_diff_bytes / 1024,
        peak / (1024 * 1024 * 1024),
    )
    logging.info(
        "[process_id=%s][name=%s] Total new allocations: %s",
        process_index,
        name,
        total_new_allocs,
    )

    logging.info(
        "[process_id=%s][name=%s] Top %d line-item memory differences:",
        process_index,
        name,
        top_n,
    )
    for index, stat in enumerate(stats[:top_n]):
      size_diff_kb = stat.size_diff / 1024
      if size_diff_kb == 0 and stat.count_diff == 0:
        continue

      frame = stat.traceback[0]
      filename = os.path.basename(frame.filename)
      logging.info(
          "[process_id=%s][name=%s]   #%d: %+.2f KiB, %+d new allocs | at"
          " %s:%s",
          process_index,
          name,
          index + 1,
          size_diff_kb,
          stat.count_diff,
          filename,
          frame.lineno,
      )

      # Get the line from the source file
      line = linecache.getline(frame.filename, frame.lineno).strip()
      if line:
        logging.info("      >> %s", line)

    logging.info(
        "[process_id=%s][name=%s] --- End of snapshot comparison ---",
        process_index,
        name,
    )


class TensorstoreMetric(BaseMetric):
  """Measures tensorstore metrics."""

  OMIT_REGISTRY_KEY_PREFIX = True
  _start_metrics: dict[str, dict[str, Any]]

  def start(self):
    super().start()
    self._start_metrics = self._collect_metrics()

  def stop(self) -> dict[str, tuple[Any, str]]:
    results = super().stop()
    end_metrics = self._collect_metrics()
    diff = self._diff_metrics(self._start_metrics, end_metrics)
    logging.info(
        "[process_id=%s] Finished metric: %s, num_diffs=%d",
        multihost.get_process_index(),
        self.name,
        len(diff),
    )
    # log all start metrics
    for key, values in self._start_metrics.items():
      logging.info(
          "TensorstoreMetric[%s] start for %s: %s", self.name, key, values
      )
    logging.info("----------------------------------------------------------")

    # log all end metrics
    for key, values in end_metrics.items():
      logging.info(
          "TensorstoreMetric[%s] end for %s: %s", self.name, key, values
      )
    logging.info("----------------------------------------------------------")

    for key, values in diff.items():
      logging.info(
          "TensorstoreMetric[%s] diff for %s: %s", self.name, key, values
      )

    # Log the number of metrics that have a non-zero diff.
    results["6_tensorstore/diff_count"] = (len(diff), "count")
    return results

  def _collect_metrics(self) -> dict[str, dict[str, Any]]:
    """Collects tensorstore metrics for interested metrics."""

    interested_metric_paths = ["/tensorstore", "/mallocz", "/tcmalloc/"]
    metrics_list = []
    for path in interested_metric_paths:
      try:
        metrics_list += ts.experimental_collect_matching_metrics(path)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "Failed to collect tensorstore metrics for path %s: %s", path, e
        )
    metrics_dict = {}
    for m in metrics_list:
      if m and "name" in m and m.get("values"):
        # For now, only consider metrics with a single value entry that
        # contains 'value' or 'count'.
        if len(m["values"]) == 1:
          metrics_dict[m["name"]] = m["values"][0]
    return metrics_dict

  def _diff_metrics(
      self,
      start_metrics: dict[str, dict[str, Any]],
      end_metrics: dict[str, dict[str, Any]],
  ) -> dict[str, dict[str, Any]]:
    """Diffs two dictionaries of metrics."""
    diff_metrics = {}
    all_keys = set(start_metrics.keys()) | set(end_metrics.keys())
    for key in all_keys:
      start_vals = start_metrics.get(key, {})
      end_vals = end_metrics.get(key, {})

      diff = {}
      if "value" in end_vals or "value" in start_vals:
        start_v = start_vals.get("value", 0)
        end_v = end_vals.get("value", 0)
        if isinstance(start_v, (int, float)) and isinstance(
            end_v, (int, float)
        ):
          val_diff = end_v - start_v
          if val_diff != 0:
            diff["value"] = val_diff

      if "count" in end_vals or "count" in start_vals:
        start_c = start_vals.get("count", 0)
        end_c = end_vals.get("count", 0)
        if isinstance(start_c, (int, float)) and isinstance(
            end_c, (int, float)
        ):
          count_diff = end_c - start_c
          if count_diff != 0:
            diff["count"] = count_diff

      if diff:
        diff_metrics[key] = diff

    return diff_metrics


# Registry of available metric types
METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
    "time": TimeMetric,
    "rss": RssMetric,
    "tracemalloc": TracemallocMetric,
    "tensorstore": TensorstoreMetric,
}

DEFAULT_METRICS = ["time"]


@dataclasses.dataclass
class Metrics:
  """Container and manager for all metric results from a profiling block."""

  results: MutableMapping[str, tuple[Any, str]] = dataclasses.field(
      default_factory=dict
  )
  name: str = ""

  def _add_results(
      self,
      metric_name: str,
      metric_key: str,
      metric_results: dict[str, tuple[Any, str]],
  ):
    for key, (value, unit) in metric_results.items():
      if metric_key:
        full_key = f"{metric_name}_{metric_key}_{key}"
      else:
        full_key = f"{metric_name}_{key}"
      self.results[full_key] = (value, unit)

  @contextlib.contextmanager
  def measure(self, operation_name: str, metric_keys: list[str] | None = None):
    """Context manager to measure a block of code with the specified metrics.

    Args:
      operation_name: The name of the operation to measure.
      metric_keys: The keys of the metrics to measure. If None, the default
        metrics (time) will be measured.
    """
    if metric_keys is None:
      metric_keys = DEFAULT_METRICS

    collector = _MetricsCollector(self, operation_name, metric_keys)
    with collector:
      yield

  def report(self):
    """Logs a formatted report of all collected metrics."""
    report_lines = []
    report_lines.append(
        f"---[process_id={multihost.get_process_index()}] {self.name} Metrics"
        " Report ---"
    )
    if not self.results:
      report_lines.append(
          f"[process_id={multihost.get_process_index()}] No metrics recorded."
      )
    else:
      for name, (value, unit) in sorted(self.results.items()):
        if isinstance(value, float):
          report_lines.append(f"{name}: {value:.4f} {unit}")
        else:
          report_lines.append(f"{name}: {value} {unit}")
    report_lines.append("----------------------")
    logging.info("\n".join(report_lines))


class _MetricsCollector:
  """Internal context manager to collect specified metrics."""

  def __init__(
      self, metrics_obj: Metrics, operation_name: str, metric_keys: list[str]
  ):
    self.metrics_obj = metrics_obj
    self.operation_name = operation_name
    self._metrics: dict[str, BaseMetric] = {}

    for key in metric_keys:
      if key in METRIC_REGISTRY:
        metric_class = METRIC_REGISTRY[key]
        self._metrics[key] = metric_class(operation_name)
      else:
        logging.warning("Unknown metric key: %s", key)

  def __enter__(self):
    for metric in self._metrics.values():
      metric.start()
    return self

  def __exit__(self, *exc):
    for key, metric in self._metrics.items():
      try:
        metric_results = metric.stop()
        tag_key = "" if metric.OMIT_REGISTRY_KEY_PREFIX else key
        self.metrics_obj._add_results(metric.name, tag_key, metric_results)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error stopping metric %s: %s", metric.name, e)


################################################################################
# Aggregation and Reporting
################################################################################


def _options_to_hparams(options: Any) -> dict[str, bool | int | float | str]:
  """Flattens a benchmark options object into a TB HParams-acceptable dict.

  HParams values must be primitives (bool / int / float / str). Anything else
  (None, list, tuple, nested) is rendered via str() so the run still appears
  in the Parallel Coordinates view rather than getting dropped.

  Args:
    options: A dataclass instance or dict of benchmark options to flatten;
      anything else yields an empty dict.

  Returns:
    A dict of primitive HParam values keyed by option name.
  """
  if dataclasses.is_dataclass(options):
    raw = dataclasses.asdict(options)
  elif isinstance(options, dict):
    raw = dict(options)
  else:
    return {}
  out: dict[str, bool | int | float | str] = {}
  for k, v in raw.items():
    if isinstance(v, (bool, int, float, str)):
      out[k] = v
    else:
      out[k] = str(v)
  return out


def _summary_aggregates(
    per_host_matrix: np.ndarray, keys: list[str]
) -> dict[str, dict[str, float]]:
  """Computes per-metric aggregates across hosts.

  Returns max / min / mean for every key, plus p50 and p99 when more than
  one host's value is present. Max is the MLPerf-honest headline number
  (slowest rank wins for time-shaped metrics; smallest rank wins for
  throughput — callers translate per metric).

  Hosts that didn't report a given metric appear as NaN in that column and
  are skipped from that metric's aggregate (so "primary-only" events like
  metadata_write_s still report a meaningful value even though most hosts
  never fire the event).
  """
  if per_host_matrix.size == 0 or len(keys) == 0:
    return {}
  out: dict[str, dict[str, float]] = {}
  for j, key in enumerate(keys):
    column = per_host_matrix[:, j]
    column = column[~np.isnan(column)]
    if column.size == 0:
      continue
    entry = {
        "max": float(np.max(column)),
        "min": float(np.min(column)),
        "mean": float(np.mean(column)),
    }
    if column.size > 1:
      entry["p50"] = float(np.percentile(column, 50))
      entry["p99"] = float(np.percentile(column, 99))
    out[key] = entry
  return out


_SCORECARD_HEADLINE_KEYS: tuple[tuple[str, str, str], ...] = (
    # Keys carry the measure() operation prefix (save_blocking_, save_background_,
    # load_) that `_add_results` splices in front of the namespaced metric key.
    # (aggregate key, label, stat to use as the headline)
    ("save_blocking_4_throughput/save_blocking_gbps",
     "Save blocking throughput (max GiB/s)", "max"),
    ("save_background_4_throughput/save_total_gbps",
     "Save total throughput (max GiB/s)", "max"),
    ("load_4_throughput/load_total_gbps",
     "Load throughput (max GiB/s)", "max"),
    ("load_4_throughput/load_per_host_gbps",
     "Load per-host throughput (max GiB/s)", "max"),
    ("save_background_5_inventory/save_total_gb",
     "Save total per host (GiB)", "max"),
    ("load_5_inventory/load_total_gb",
     "Load total per host (GiB)", "max"),
    ("save_blocking_2_save_breakdown/blocking_async_s",
     "Save blocking (slowest host, s)", "max"),
    ("load_3_load_breakdown/blocking_s",
     "Load blocking (slowest host, s)", "max"),
    ("save_blocking_7_overhead/sync_global_devices_s",
     "Sync-barrier overhead (slowest host, s)", "max"),
)


def _render_scorecard_markdown(
    benchmark_name: str,
    aggregates: dict[str, dict[str, float]],
    inventory: "Any | None",
    manifest: "Any | None",
) -> str:
  """Renders the per-benchmark scorecard.

  Headline metrics come from the cross-host aggregate dict
  (`{metric_key → {stat_name → value}}`) keyed off the curated list above.
  Inventory + manifest sections are included only when provided so the card
  stays compact for benchmarks that opted out of those captures.
  """
  lines = [f"## {benchmark_name} — scorecard", ""]

  headline_rows = []
  for agg_key, label, stat in _SCORECARD_HEADLINE_KEYS:
    if agg_key in aggregates and stat in aggregates[agg_key]:
      headline_rows.append((label, aggregates[agg_key][stat]))

  if headline_rows:
    lines.extend(["### Headline numbers", ""])
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for label, value in headline_rows:
      lines.append(f"| {label} | {value:.4f} |")
    lines.append("")

  if inventory is not None:
    lines.extend(["### Inventory", ""])
    lines.append("| field | value |")
    lines.append("|---|---:|")
    total_gb = inventory.total_bytes / (1024 ** 3)
    lines.append(f"| total bytes | {total_gb:.2f} GiB |")
    lines.append(f"| file count | {inventory.file_count:,} |")
    small_pct = inventory.small_file_pct * 100
    canary = "✓" if small_pct < 10 else "⚠ chunk_byte_size too small?"
    lines.append(f"| small files <1 MiB | {small_pct:.1f}% {canary} |")
    if inventory.largest_file_bytes > 0:
      lines.append(
          f"| largest file | {inventory.largest_file_bytes / (1024**2):.2f} MiB |"
      )
    if inventory.format:
      fmt_str = ", ".join(f"{k}={v}" for k, v in sorted(inventory.format.items()))
      lines.append(f"| format breakdown | {fmt_str} |")
    lines.append("")

  if manifest is not None:
    lines.append(manifest.as_markdown())

  return "\n".join(lines)


def _render_configuration_markdown(
    benchmark_name: str,
    benchmark_options: dict[str, Any] | None,
    checkpoint_config: dict[str, Any] | None,
) -> str:
  """Renders the run configuration as readable markdown.

  Options + checkpoint_config become two field/value tables; any nested
  dict in checkpoint_config (typically `spec`) is split out into its own
  fenced-JSON block. Replaces the single-line `json.dumps` blob the
  Text-tab card used to show.

  Args:
    benchmark_name: Title rendered as the top-level `##` heading.
    benchmark_options: Flat option name/value pairs, or None to omit the table.
    checkpoint_config: Checkpoint config; scalar entries form a table and each
      nested dict becomes its own fenced-JSON block.

  Returns:
    The configuration rendered as a markdown string.
  """
  lines = [f"## {benchmark_name}", ""]

  def _table(title: str, items: list[tuple[str, Any]]) -> None:
    lines.append(f"### {title}")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    for k, v in items:
      lines.append(f"| `{k}` | `{v}` |")
    lines.append("")

  if benchmark_options:
    _table(
        "Benchmark options",
        [(k, v) for k, v in sorted(benchmark_options.items())],
    )

  if checkpoint_config:
    scalar_items = []
    nested_items = []
    for k, v in sorted(checkpoint_config.items()):
      if isinstance(v, dict):
        nested_items.append((k, v))
      else:
        scalar_items.append((k, v))
    if scalar_items:
      _table("Checkpoint config", scalar_items)
    for k, v in nested_items:
      lines.append(f"### Checkpoint config — `{k}`")
      lines.append("")
      lines.append("```json")
      lines.append(json.dumps(v, indent=2, sort_keys=True))
      lines.append("```")
      lines.append("")
  return "\n".join(lines)


# TODO(b/519204863): Move rendering and related changes to a separate file.
def _render_aggregated_metrics_markdown(
    benchmark_name: str,
    aggregated_stats_dict: dict[str, "AggregatedStats"],
    metric_units: dict[str, str],
    host_label: str | None = None,
) -> str:
  """Renders the aggregated metrics as a markdown table grouped by `/` prefix.

  TB's Text dashboard renders markdown — a proper table is dramatically
  more readable than the previous `<pre>` raw dump, and grouping by
  numbered prefix (`1_overview/`, `2_save_breakdown/`, …) mirrors the
  Scalars-view navigation so a reader can locate a metric the same way
  in both surfaces.

  Args:
    benchmark_name: Title rendered as the top-level heading.
    aggregated_stats_dict: Metric tag -> aggregated stats to tabulate.
    metric_units: Metric tag -> unit string shown alongside each value.
    host_label: When set, lands in the markdown header so a reader who
      selected multiple per-host runs can tell whose aggregates they're
      looking at.

  Returns:
    The aggregated metrics rendered as a markdown string.
  """
  if not aggregated_stats_dict:
    return "_No successful runs to aggregate._"

  groups: dict[str, list[str]] = collections.defaultdict(list)
  for key in sorted(aggregated_stats_dict):
    head, _, _ = key.partition("/")
    section = head if "/" in key else "_other_"
    groups[section].append(key)

  suffix = f" — {host_label}" if host_label else ""
  lines = [f"## {benchmark_name} — aggregated metrics{suffix}", ""]
  for section in sorted(groups):
    lines.append(f"### {section}")
    lines.append("")
    lines.append("| metric | mean | ± std | min | max | n | unit |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for key in groups[section]:
      stats = aggregated_stats_dict[key]
      unit = metric_units.get(key, "")
      leaf = key.split("/", 1)[1] if "/" in key else key
      lines.append(
          f"| `{leaf}` | {stats.mean:.4f} | {stats.std:.4f} |"
          f" {stats.min:.4f} | {stats.max:.4f} | {stats.count} | {unit} |"
      )
    lines.append("")
  return "\n".join(lines)


@dataclasses.dataclass
class AggregatedStats:
  """Statistics aggregated over multiple benchmark repetitions.

  Attributes:
    mean: Mean value.
    std: Standard deviation.
    min: Minimum value.
    max: Maximum value.
    count: Number of values aggregated.
  """

  mean: float
  std: float
  min: float
  max: float
  count: int


class MetricsManager:
  """Manages metrics aggregation and reporting for a test suite.

  This class collects metrics from multiple benchmark runs and repetitions,
  computes aggregate statistics (mean, std, min, max), generates a
  human-readable report for logging, and exports metrics to TensorBoard
  if configured.
  """

  def __init__(
      self,
      name: str,
      num_repeats: int,
      tensorboard_dir: epath.Path | None = None,
      enable_per_host_metrics: bool = True,
  ):
    """Initializes the MetricsManager.

    Args:
      name: The name of the test suite.
      num_repeats: The number of repetitions for each benchmark configuration.
      tensorboard_dir: The directory to write TensorBoard events to. If None,
        metrics will not be written to TensorBoard during the run.
      enable_per_host_metrics: When True, every process opens its own writer
        at <tensorboard_dir>/<benchmark>/host_<idx>/ so per-host scalars are
        visible in the TB Scalars view as sibling runs. When False, only the
        primary host writes (legacy behavior).
    """
    self._name = name
    self._num_repeats = num_repeats
    self._runs: dict[str, list[tuple[Metrics, Exception | None]]] = (
        collections.defaultdict(list)
    )
    self._benchmark_options: dict[str, Any] = {}
    self._checkpoint_configs: dict[str, Any] = {}
    self._tensorboard_dir = tensorboard_dir
    self._enable_per_host_metrics = enable_per_host_metrics
    self._writers: dict[str, Any] = {}
    # Inventory is suite-level (one per benchmark; the first non-None we see
    # wins) — it's the post-save filesystem walk from Benchmark.run, only
    # collected on the primary host.
    self._inventories: dict[str, Any] = {}
    # Suite-level environment snapshot. Captured once at suite construction
    # so the manifest matches the run that gets reported, not the latest
    # state of the machine when generate_report is called.
    self._suite_run_manifest = run_manifest_lib.capture_run_manifest()

  def add_result(
      self,
      benchmark_name: str,
      metrics: Metrics,
      *,
      benchmark_options: Any | None = None,
      checkpoint_config: Any | None = None,
      error: Exception | None = None,
      inventory: Any | None = None,
  ):
    """Adds metrics from a single benchmark run/repetition.

    Args:
      benchmark_name: The name of the benchmark configuration.
      metrics: The Metrics object containing results for this run.
      benchmark_options: The BenchmarkOptions used for this run.
      checkpoint_config: The CheckpointConfig used for this run.
      error: An exception if the run failed, otherwise None.
      inventory: Optional post-save CheckpointInventory; the first non-None
        provided per benchmark wins (subsequent repeats overwrite the same
        target dir, so the inventory is invariant across repeats).
    """
    self._runs[benchmark_name].append((metrics, error))
    if benchmark_name not in self._benchmark_options:
      self._benchmark_options[benchmark_name] = benchmark_options
    if benchmark_name not in self._checkpoint_configs:
      self._checkpoint_configs[benchmark_name] = checkpoint_config
    if inventory is not None and benchmark_name not in self._inventories:
      self._inventories[benchmark_name] = inventory

    if self._tensorboard_dir:
      self._write_result_to_tensorboard(
          benchmark_name,
          metrics,
          error,
          len(self._runs[benchmark_name]) - 1,
          benchmark_options,
          checkpoint_config,
      )

  def _get_writer(self, benchmark_name: str) -> Any:
    """Gets or creates a TensorBoard writer for the given benchmark.

    When per-host metrics are enabled, every process opens its own writer
    at `<tensorboard_dir>/<benchmark>/host_<idx>/`, surfacing each host as a
    sibling TB run for the Distribution / Compare views. Otherwise only
    the primary host writes (legacy behavior).
    """
    if benchmark_name in self._writers:
      return self._writers[benchmark_name]

    host_idx = multihost.get_process_index()
    if self._enable_per_host_metrics:
      # clu's create_default_writer plants events at <logdir>/<collection>/.
      # Encoding the host suffix in collection keeps the on-disk layout flat
      # at "<tensorboard>/<benchmark>/host_<idx>/" without double-nesting.
      writer = metric_writers.create_default_writer(
          self._tensorboard_dir,
          collection=f"{benchmark_name}/host_{host_idx}",
      )
    else:
      writer = metric_writers.create_default_writer(
          self._tensorboard_dir,
          just_logging=host_idx != 0,
          collection=benchmark_name,
      )
    self._writers[benchmark_name] = writer
    return writer

  def _write_result_to_tensorboard(
      self,
      benchmark_name: str,
      metrics: Metrics,
      error: Exception | None,
      step: int,
      benchmark_options: Any | None = None,
      checkpoint_config: Any | None = None,
  ):
    """Writes a single result to TensorBoard."""
    writer = self._get_writer(benchmark_name)
    if error is None:
      for key, (value, unit) in metrics.results.items():
        # Hierarchical keys (e.g. "2_save_breakdown/blocking_s") already
        # encode the unit in their suffix; appending "_s" again gives the
        # ugly "..._blocking_s_s". Skip the suffix for those; flat legacy
        # keys keep the existing "{key}_{unit}" shape.
        if "/" in key:
          tag = key
        else:
          tag = f'{key}_{unit.replace("/", "_")}'
        if isinstance(value, (int, float)):
          writer.write_scalars(step=step, scalars={tag: value})
        else:
          writer.write_texts(step=step, texts={tag: str(value)})
    else:
      tag = "error"
      writer.write_texts(step=step, texts={tag: f"<pre>{repr(error)}</pre>"})

    # Configuration text + HParams summary are SUITE-level (identical on
    # every host), so only the primary host writes them. Otherwise every
    # per-host writer dir gets a duplicate card, cluttering the Text and
    # HParams tabs and adding N identical Parallel-Coordinates rows.
    is_primary = multihost.get_process_index() == 0
    if step == 0 and benchmark_options and is_primary:
      if dataclasses.is_dataclass(benchmark_options):
        opt_dict = dataclasses.asdict(benchmark_options)
      else:
        opt_dict = benchmark_options

      if dataclasses.is_dataclass(checkpoint_config):
        config_dict = dataclasses.asdict(checkpoint_config)
      elif isinstance(checkpoint_config, dict):
        config_dict = checkpoint_config
      else:
        config_dict = None

      writer.write_texts(
          step=0,
          texts={
              "configuration": _render_configuration_markdown(
                  benchmark_name, opt_dict, config_dict
              ),
          },
      )
      if hparams_dict := _options_to_hparams(benchmark_options):
        writer.write_hparams(hparams_dict)
    writer.flush()

  def _aggregate_metrics(
      self, results: list[tuple[Metrics, Exception | None]]
  ) -> tuple[dict[str, AggregatedStats], dict[str, str]]:
    """Computes aggregate stats (mean, std, etc.) for successful runs.

    Args:
      results: A list of (Metrics, error) tuples for a benchmark configuration.

    Returns:
      A tuple containing:
        - A dict mapping metric keys to AggregatedStats.
        - A dict mapping metric keys to their units.
    """
    metrics_collector = collections.defaultdict(list)
    metric_units = {}
    for metrics, error in results:
      if error is None:
        for key, (value, unit) in metrics.results.items():
          if isinstance(value, (int, float)):
            metrics_collector[key].append(value)
            metric_units[key] = unit

    aggregated_stats_dict = {}
    for key, values in metrics_collector.items():
      aggregated_stats_dict[key] = AggregatedStats(
          mean=np.mean(values),
          std=np.std(values),
          min=np.min(values),
          max=np.max(values),
          count=len(values),
      )
    return aggregated_stats_dict, metric_units

  def _count_runs(self) -> tuple[int, int, int]:
    total = passed = failed = 0
    for _, results in self._runs.items():
      total += len(results)
      for _, error in results:
        if error is None:
          passed += 1
        else:
          failed += 1
    return total, passed, failed

  def _format_stats_lines(
      self,
      aggregated_stats_dict: dict[str, AggregatedStats],
      metric_units: dict[str, str],
      indent: str,
  ) -> list[str]:
    """Formats aggregated stats into human-readable report lines."""
    lines = []
    for key, stats in aggregated_stats_dict.items():
      unit = metric_units[key]
      lines.append(
          f"{indent}{key}: {stats.mean:.4f} +/- {stats.std:.4f} {unit} (min:"
          f" {stats.min:.4f}, max: {stats.max:.4f}, n={stats.count})"
      )
    return lines

  def _format_aggregated_report_section(self) -> list[str]:
    """Builds the per-benchmark aggregated-metrics section of the report."""
    lines = ["\n" + "-" * 80, "--- Aggregated Metrics per Benchmark ---"]
    for benchmark_name, results in self._runs.items():
      if not results:
        continue
      lines.append(f"\nBenchmark: {benchmark_name}")
      aggregated_stats_dict, metric_units = self._aggregate_metrics(results)
      if not aggregated_stats_dict:
        lines.append("  No successful runs to aggregate.")
        continue
      lines.extend(
          self._format_stats_lines(aggregated_stats_dict, metric_units, "  ")
      )
    return lines

  def _format_failed_runs_section(self) -> list[str]:
    lines = ["\n" + "-" * 80, "--- Failed Runs ---"]
    for _, results in self._runs.items():
      for metrics, error in results:
        if error is None:
          continue
        error_repr = repr(error)
        # Limit error length to avoid flooding logs.
        if len(error_repr) > 1000:
          error_repr = error_repr[:1000] + "..."
        lines.append(f"Test: {metrics.name}, Error: {error_repr}")
    return lines

  def _write_aggregated_to_tensorboard(self) -> None:
    """Writes each benchmark's aggregated metrics to TensorBoard as text."""
    logging.info("Writing aggregated metrics to TensorBoard...")
    host_label = (
        f"host_{multihost.get_process_index()}"
        if self._enable_per_host_metrics
        else None
    )
    for benchmark_name, results in self._runs.items():
      writer = self._get_writer(benchmark_name)
      aggregated_stats_dict, metric_units = self._aggregate_metrics(results)
      aggregated_metrics_str = _render_aggregated_metrics_markdown(
          benchmark_name,
          aggregated_stats_dict,
          metric_units,
          host_label=host_label,
      )
      writer.write_texts(
          step=0,
          texts={"aggregated_metrics": aggregated_metrics_str},
      )
      writer.flush()
      writer.close()

      # Cross-host gather + summary writer. Lives outside the per-host
      # writer scope above so the aggregates land in a sibling __summary__
      # run that TB can compare against any individual host run.
      self._aggregate_and_write_summary(benchmark_name, results)
    # Clear writers after closing to prevent reuse of closed writers if called
    # again.
    self._writers.clear()
    logging.info("Finished writing metrics to TensorBoard.")

  def generate_report(self) -> None:
    """Generates a final string report containing aggregated metrics.

    And exports aggregated metrics to TensorBoard if configured.
    """
    title = f" Test Suite Report: {self._name} "
    report_lines = [f"\n{title:=^80}"]
    total_runs, passed_runs, failed_runs = self._count_runs()
    report_lines.append(f"Total benchmark configurations: {len(self._runs)}")
    report_lines.append(
        f"Total runs ({self._num_repeats} repeats): {total_runs}, Passed:"
        f" {passed_runs}, Failed: {failed_runs}"
    )
    if self._num_repeats > 1:
      report_lines.extend(self._format_aggregated_report_section())
    if failed_runs > 0:
      report_lines.extend(self._format_failed_runs_section())
    report_lines.append("\n" + "=" * 80)
    logging.info("\n".join(report_lines))
    if self._tensorboard_dir:
      self._write_aggregated_to_tensorboard()

  def _aggregate_and_write_summary(
      self,
      benchmark_name: str,
      results: list[tuple[Metrics, Exception | None]],
  ) -> None:
    """Gathers per-host metric values across processes and writes summary.

    Each host contributes its mean-across-repeats value for every numeric
    metric key. `process_allgather` builds a (host_count, metric_count)
    matrix; the primary host computes max/min/mean (and p50/p99 when more
    than one host is present) plus per-metric histograms and writes them
    to <tensorboard_dir>/<benchmark>/__summary__/.

    Honest at scale: the "max" entry is the MLPerf-shape headline number
    (slowest rank wins for time, smallest rank for throughput — callers
    pick the relevant column per metric).
    """
    if not self._enable_per_host_metrics or self._tensorboard_dir is None:
      return

    metrics_collector: dict[str, list[float]] = collections.defaultdict(list)
    for m, error in results:
      if error is None:
        for k, (v, _unit) in m.results.items():
          if isinstance(v, (int, float)):
            metrics_collector[k].append(float(v))

    # Do not early-return when this host has no successful metrics: every
    # host must still reach the sync_global_processes barriers below, or a
    # host that diverged (all runs errored for this benchmark while peers
    # succeeded) would skip the collective and hang the rest of the run. An
    # empty sidecar is harmless — the primary unions keys across hosts.
    canonical_keys = sorted(metrics_collector)
    per_host_means = np.array(
        [float(np.mean(metrics_collector[k])) for k in canonical_keys],
        dtype=np.float64,
    )

    # Filesystem-based gather: each host drops its per-host means as a JSON
    # sidecar under its per-host writer dir; primary reads them all back.
    # This avoids jax.distributed allgather (whose Gloo backend has been
    # flaky around process-shutdown in this docker harness) and works on
    # any shared fs the per-host writers already use — local mount, GCS,
    # NFS, etc.
    host_idx = multihost.get_process_index()
    host_dir = self._tensorboard_dir / benchmark_name / f"host_{host_idx}"
    try:
      host_dir.mkdir(parents=True, exist_ok=True)
      (host_dir / "_per_host_means.json").write_text(
          json.dumps({
              "keys": canonical_keys,
              "means": per_host_means.tolist(),
          })
      )
    except OSError as e:
      logging.warning("Failed to write per-host means sidecar: %s", e)

    multihost.sync_global_processes(f"metrics:summary:{benchmark_name}")

    if host_idx == 0:
      benchmark_dir = self._tensorboard_dir / benchmark_name
      per_host_dicts: list[dict[str, float]] = []
      for host_subdir in sorted(benchmark_dir.iterdir()):
        if not host_subdir.name.startswith("host_"):
          continue
        sidecar = host_subdir / "_per_host_means.json"
        if not sidecar.exists():
          continue
        data = json.loads(sidecar.read_text())
        per_host_dicts.append(dict(zip(data["keys"], data["means"])))

      if per_host_dicts:
        # Union the keys across hosts; missing-on-some-hosts becomes NaN
        # so _summary_aggregates can still compute the per-key max/p99 from
        # the hosts that did report (e.g. metadata_write_s is primary-only
        # but still meaningful as a headline number).
        union_keys = sorted(set().union(*(d.keys() for d in per_host_dicts)))
        all_hosts_arr = np.full(
            (len(per_host_dicts), len(union_keys)), np.nan, dtype=np.float64
        )
        for i, d in enumerate(per_host_dicts):
          for j, k in enumerate(union_keys):
            if k in d:
              all_hosts_arr[i, j] = d[k]

        aggregates = _summary_aggregates(all_hosts_arr, union_keys)
        if aggregates:
          summary_writer = metric_writers.create_default_writer(
              self._tensorboard_dir,
              collection=f"{benchmark_name}/__summary__",
          )
          try:
            # Only the aggregate scalars (max/min/mean/p50/p99) land in the
            # summary. A separate write_histograms call would surface in
            # TB's Histograms / Distributions tabs but those views are
            # designed to plot percentile bands or histograms ACROSS STEPS
            # — at step 0 only they collapse to a degenerate line and look
            # broken to anyone clicking the tab. Users who want a per-host
            # visual comparison select the host_N runs in the left rail
            # and overlay them in Time Series.
            for key, stats in aggregates.items():
              scalars = {
                  f"{key}_{stat}": value for stat, value in stats.items()
              }
              summary_writer.write_scalars(step=0, scalars=scalars)

            # Scorecard with headline numbers + inventory + manifest. The
            # scorecard is the thing people screenshot for PRs/reports;
            # everything else is plumbing.
            inventory = self._inventories.get(benchmark_name)
            manifest = self._suite_run_manifest
            scorecard_md = _render_scorecard_markdown(
                benchmark_name, aggregates, inventory, manifest
            )
            summary_writer.write_texts(
                step=0, texts={"scorecard": scorecard_md}
            )

            summary_writer.flush()
          finally:
            summary_writer.close()

    # Final barrier so non-primaries don't exit while primary is still
    # writing the summary card. Without this, the coordinator marks the
    # early-exiting tasks as gone and the shutdown barrier fails.
    multihost.sync_global_processes(f"metrics:summary-done:{benchmark_name}")
