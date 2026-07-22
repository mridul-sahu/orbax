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

"""Ordered composition of operations into a resolvable plan.

A pipeline resolves in a single pass: it builds a manifest from source metadata,
folds the operations over it, then validates the result against the abstract
target. Validation is the load minus the bytes, so a preview is cheap enough to
keep in a unit test.
"""

import dataclasses
import math
from typing import Any

import jax
import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import manifest as manifest_lib
from orbax.checkpoint.experimental.v1._src.surgery import operations as operations_lib
from orbax.checkpoint.experimental.v1._src.surgery import report as report_lib
from orbax.checkpoint.experimental.v1._src.surgery import trees

Manifest = manifest_lib.Manifest
LeafRef = manifest_lib.LeafRef
VirtualLeaf = manifest_lib.VirtualLeaf
PlanReport = report_lib.PlanReport
Op = operations_lib.Op

# The namespace a single source tree is filed under. Multi-source merges name
# their own namespaces through `take`.
DEFAULT_SOURCE = "source"


def _make_read(spec: Any) -> manifest_lib.ReadFn | None:
  """Returns a sub-range reader for `spec`, or `None` for a metadata leaf.

  A `StoredArray` reads each requested sub-range straight from storage. An
  in-memory array is brought to host once and sliced per read. A metadata or
  abstract leaf carries no data and returns `None`.

  Args:
    spec: A source leaf: a `StoredArray`, an array, an abstract leaf, or a
      metadata object.

  Returns:
    A reader, or `None` when `spec` carries no array data.
  """
  if isinstance(spec, manifest_lib.StoredArray):
    return spec.read
  if isinstance(spec, (np.ndarray, jax.Array)):
    host = np.asarray(spec)
    return lambda region, host=host: host[region.slices]
  return None


def _leaf_ref(namespace: str, key: str, spec: Any) -> LeafRef:
  return LeafRef(
      source=namespace,
      key=key,
      shape=tuple(spec.shape),
      dtype=np.dtype(spec.dtype),
      read=_make_read(spec),
  )


def _zeros_init(
    shape: manifest_lib.Shape, dtype: np.dtype
) -> np.ndarray:
  return np.zeros(shape, dtype=dtype)


def _check_coverage(key: str, leaf: VirtualLeaf, report: PlanReport) -> None:
  pair = manifest_lib.overlapping_assignments(leaf.assignments)
  if pair is not None:
    report.add_error(
        report_lib.OVERLAPPING_ASSIGNMENTS,
        key,
        f"assignments {pair[0]} and {pair[1]} claim overlapping regions",
    )
  covered = manifest_lib.covered_size(leaf.assignments)
  full = math.prod(leaf.shape)
  if leaf.init is None and covered != full:
    report.add_error(
        report_lib.UNCOVERED_TARGET,
        key,
        f"assignments cover {covered} of {full} elements with no initializer",
    )


def _check_shape_dtype(
    key: str, leaf: manifest_lib.Leaf, target_spec: Any, report: PlanReport
) -> None:
  if tuple(leaf.shape) != tuple(target_spec.shape):
    report.add_error(
        report_lib.SHAPE_MISMATCH,
        key,
        f"produced shape {tuple(leaf.shape)} != target {tuple(target_spec.shape)}",
    )
  if np.dtype(leaf.dtype) != np.dtype(target_spec.dtype):
    report.add_error(
        report_lib.DTYPE_MISMATCH,
        key,
        f"produced dtype {np.dtype(leaf.dtype)} != target"
        f" {np.dtype(target_spec.dtype)}",
    )


def _validate(
    manifest: Manifest,
    target_flat: dict[str, Any] | None,
    ctx: operations_lib.ResolveContext,
) -> None:
  """Runs coverage checks and, when a target is given, conformance checks."""
  for key, leaf in manifest.items():
    if isinstance(leaf, VirtualLeaf):
      _check_coverage(key, leaf, ctx.report)

  if target_flat is None:
    return

  for tkey, tspec in target_flat.items():
    if tkey in manifest:
      _check_shape_dtype(tkey, manifest[tkey], tspec, ctx.report)
    elif ctx.on_missing == "init_from_target":
      ctx.report.filled[tkey] = "init_from_target"
      manifest[tkey] = VirtualLeaf(
          shape=tuple(tspec.shape),
          dtype=np.dtype(tspec.dtype),
          assignments=(),
          init=_zeros_init,
      )
    else:
      ctx.report.add_error(
          report_lib.MISSING_SOURCE,
          tkey,
          f"target leaf {tkey!r} is not produced by the pipeline",
      )

  for mkey in manifest:
    if mkey not in target_flat:
      ctx.report.add_error(
          report_lib.UNEXPECTED_KEY,
          mkey,
          f"{mkey!r} is produced but has no place in the target",
      )


@dataclasses.dataclass
class ResolvedPlan:
  """A fully resolved manifest and its account.

  Attributes:
    manifest: The final manifest, one leaf per target key.
    report: The account of what the plan does, including any errors.
  """

  manifest: Manifest
  report: PlanReport


@dataclasses.dataclass(frozen=True)
class Plan:
  """An ordered set of operations plus the missing-target policy.

  Attributes:
    ops: The operations, applied in order.
    on_missing: Either "error" or "init_from_target".
  """

  ops: tuple[Op, ...]
  on_missing: str

  def _resolve_refs(
      self,
      source_refs: dict[str, dict[str, LeafRef]],
      initial_namespace: str | None,
      target: Any,
  ) -> ResolvedPlan:
    """Folds the operations over a manifest built from source references.

    Args:
      source_refs: Per-namespace leaf references, keyed by namespace then key.
      initial_namespace: The namespace whose references seed the manifest, or
        `None` to start empty (a multi-source merge populated by `take`).
      target: The abstract target tree, or `None`.

    Returns:
      The resolved manifest and its report.
    """
    ctx = operations_lib.ResolveContext(
        on_missing=self.on_missing,
        report=PlanReport(),
        source_refs=source_refs,
    )
    if initial_namespace is not None:
      manifest: Manifest = dict(source_refs[initial_namespace])
    else:
      manifest = {}
    for op in self.ops:
      manifest = op(manifest, ctx)
    target_flat = trees.flatten(target) if target is not None else None
    _validate(manifest, target_flat, ctx)
    return ResolvedPlan(manifest=manifest, report=ctx.report)

  def resolve(self, source: Any, target: Any = None) -> ResolvedPlan:
    """Folds the operations over a manifest built from a single `source`.

    Args:
      source: A single source tree of leaves, arrays for execution or metadata
        for a preview.
      target: The abstract target tree, or `None` to skip conformance checks.

    Returns:
      The resolved manifest and its report.
    """
    refs = {
        key: _leaf_ref(DEFAULT_SOURCE, key, spec)
        for key, spec in trees.flatten(source).items()
    }
    return self._resolve_refs({DEFAULT_SOURCE: refs}, DEFAULT_SOURCE, target)

  def resolve_sources(
      self, sources: dict[str, Any], target: Any = None
  ) -> ResolvedPlan:
    """Folds the operations over several named sources.

    The manifest starts empty; `take` imports keys from the named sources.

    Args:
      sources: A mapping from namespace to a source tree of leaves.
      target: The abstract target tree, or `None`.

    Returns:
      The resolved manifest and its report.
    """
    source_refs = {
        namespace: {
            key: _leaf_ref(namespace, key, spec)
            for key, spec in trees.flatten(source).items()
        }
        for namespace, source in sources.items()
    }
    return self._resolve_refs(source_refs, None, target)

  def preview(self, source: Any, target: Any = None) -> PlanReport:
    """Resolves the plan and returns its report without reading array data.

    Args:
      source: Source metadata (for example from `ocp.metadata`).
      target: The abstract target tree, or `None`.

    Returns:
      The plan report.
    """
    return self.resolve(source, target).report


def pipeline(*ops: Op, on_missing: str = "error") -> Plan:
  """Builds a plan from an ordered set of operations.

  Args:
    *ops: Operations applied left to right; each sees the key space its
      predecessor produced.
    on_missing: How to treat a target leaf the pipeline does not produce,
      either "error" or "init_from_target".

  Returns:
    The plan.
  """
  if on_missing not in ("error", "init_from_target"):
    raise ValueError(
        f"on_missing must be 'error' or 'init_from_target', got {on_missing!r}"
    )
  return Plan(ops=tuple(ops), on_missing=on_missing)


def preview(plan: Plan, source: Any, target: Any = None) -> PlanReport:
  """Convenience wrapper for `plan.preview`.

  Args:
    plan: The plan to resolve.
    source: Source metadata or tree.
    target: The abstract target tree, or `None`.

  Returns:
    The plan report.
  """
  return plan.preview(source, target)
