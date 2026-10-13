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

"""Fills shards of a virtual leaf, every box in parallel, through a backend.

A backend decides where a box's values are computed. `HostBackend` lays them
out in host memory with NumPy, which suits CPU targets. `DeviceBackend` sends
the fetched bytes to the target device in stored order and lays them out there
with XLA, which suits accelerators. The default picks by the target device's
platform; a caller may pass either, or its own.
"""

import collections
from collections.abc import Callable, Iterable, Mapping, Sequence
import concurrent.futures
import os
import threading
import time
from typing import Any, Protocol

import jax
import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import boxes
from orbax.checkpoint.experimental.v1._src.surgery import device as device_lib
from orbax.checkpoint.experimental.v1._src.surgery import host
from orbax.checkpoint.experimental.v1._src.surgery import reads

Region = boxes.Region
VirtualLeaf = boxes.VirtualLeaf

# NumPy copies, TensorStore reads and JAX dispatch release the GIL, so boxes
# fill on a thread pool. The cap bounds how many reads are in flight, and with
# them the host memory held by blocks not yet consumed.
_MAX_FILL_WORKERS = min(16, os.cpu_count() or 1)


class Backend(Protocol):
  """Computes box values into one shard.

  `put` is called once per box overlapping the shard, possibly on several
  threads at once for the boxes of one shard, and `finish` once after the last.
  """

  def start(
      self, region: Region, dtype: np.dtype, device: jax.Device | None
  ) -> Any:
    """Returns the state of a new shard holding `region` of the target."""

  def put(
      self,
      shard: Any,
      box: boxes.Assignment,
      hit: Region,
      fetch: reads.FetchFn,
  ) -> None:
    """Computes `box` over `hit`, its part inside the shard, into the shard."""

  def finish(self, shard: Any) -> np.ndarray | jax.Array:
    """Returns the filled shard, on host or on its device."""


_HOST = host.HostBackend()
_DEVICE = device_lib.DeviceBackend()


def default_backend(device: jax.Device | None) -> Backend:
  """Returns the host backend for CPU targets and the device backend otherwise."""
  if device is None or device.platform == "cpu":
    return _HOST
  return _DEVICE


def _overlaps(
    leaf: VirtualLeaf, region: Region
) -> list[tuple[boxes.Assignment, Region]]:
  """Returns each box of `leaf` that overlaps `region`, with the overlap."""
  hits = []
  for box in leaf.assignments:
    hit = region.intersect(box.target_region)
    if hit is not None:
      hits.append((box, hit))
  return hits


def fill(
    leaf: VirtualLeaf,
    shards: Mapping[Region, jax.Device | None],
    on_filled: Callable[[Region, np.ndarray | jax.Array], None],
    *,
    backend_for: Callable[[jax.Device | None], Backend] = default_backend,
    stats: reads.LoadStats | None = None,
) -> None:
  """Fills each shard of `leaf`, every box of every shard in parallel.

  The boxes of a shard share one `ReadBatch`, so nearby reads of one source
  merge. A shard is handed to `on_filled` as soon as its last box lands, so
  its copy to a device can start while other boxes are still being read.

  Args:
    leaf: The virtual leaf to fill.
    shards: Each distinct part of the target, and the device it is built on,
      or `None` for host memory.
    on_filled: Receives each region and its filled shard.
    backend_for: Chooses the backend for a target device.
    stats: Collects read and host timings, if given.
  """
  states: dict[Region, tuple[Backend, Any]] = {}
  batches: dict[Region, reads.ReadBatch] = {}
  tasks: list[tuple[Region, boxes.Assignment, Region]] = []
  for region, device in shards.items():
    backend = backend_for(device)
    states[region] = (backend, backend.start(region, leaf.dtype, device))
    hits = [(region, box, hit) for box, hit in _overlaps(leaf, region)]
    batches[region] = reads.ReadBatch(
        (r for _, box, hit in hits for r in reads.read_requests(box, hit)),
        stats,
    )
    tasks.extend(hits)

  def finish(region: Region) -> None:
    backend, state = states[region]
    on_filled(region, backend.finish(state))

  remaining = collections.Counter(region for region, _, _ in tasks)
  for region in states:
    if not remaining[region]:
      finish(region)

  def run(region: Region, box: boxes.Assignment, hit: Region):
    began = time.perf_counter()
    backend, state = states[region]
    backend.put(state, box, hit, batches[region].fetch)
    return region, time.perf_counter() - began

  read_before = stats.read_seconds if stats is not None else 0.0
  busy = 0.0
  if tasks:
    workers = min(len(tasks), _MAX_FILL_WORKERS)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
      futures = [pool.submit(run, *task) for task in tasks]
      for future in concurrent.futures.as_completed(futures):
        region, seconds = future.result()
        busy += seconds
        remaining[region] -= 1
        if not remaining[region]:
          finish(region)
  if stats is not None:
    stats.host_seconds += busy - (stats.read_seconds - read_before)
    stats.shards += len(states)


def assemble_host(leaf: VirtualLeaf) -> np.ndarray:
  """Assembles a whole virtual leaf in one host buffer."""
  out = []
  fill(
      leaf,
      {Region.full(leaf.shape): None},
      lambda _, buf: out.append(buf),
      backend_for=lambda _: _HOST,
  )
  return out[0]


def derive(
    name: str,
    inputs: Sequence[VirtualLeaf],
    fn: Callable[..., Iterable[Any]],
    outputs: Sequence[tuple[boxes.Shape, np.dtype]],
) -> list[VirtualLeaf]:
  """Exposes values computed from whole inputs as derived sources.

  `fn` runs at most once, on the first read of any output, over the assembled
  inputs. Shards fill on several threads, so a lock makes the first reader
  compute and the rest reuse its results.

  Args:
    name: Names the derived sources, which are keyed `<name>.<output index>`.
    inputs: The leaves `fn` receives, assembled in full.
    fn: Maps the input arrays to one array per output.
    outputs: The shape and dtype of each output.

  Returns:
    One leaf per output, each a single view of its derived source.
  """
  stored = {
      id(ref): ref
      for leaf in inputs
      for box in leaf.assignments
      for ref in box.refs()
  }
  readable = all(ref.read is not None for ref in stored.values())
  lock = threading.Lock()
  results: list[np.ndarray] = []

  def run() -> list[np.ndarray]:
    with lock:
      if not results:
        arrays = fn(*(assemble_host(leaf) for leaf in inputs))
        results.extend(
            np.asarray(a).astype(dtype, copy=False)
            for a, (_, dtype) in zip(arrays, outputs)
        )
    return results

  leaves = []
  for k, (shape, dtype) in enumerate(outputs):
    ref = boxes.DerivedRef(
        source=boxes.DERIVED_SOURCE,
        key=f"{name}.{k}",
        shape=tuple(shape),
        dtype=np.dtype(dtype),
        read=(lambda region, k=k: run()[k][region.slices])
        if readable
        else None,
        inputs=tuple(stored.values()),
    )
    leaves.append(boxes.as_virtual(ref))
  return leaves
