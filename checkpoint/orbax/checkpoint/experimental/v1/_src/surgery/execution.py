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

"""Executes a resolved plan as a sharded read.

Execution inverts the usual direction of control: each device shard of each
target array asks which source regions fall inside it, reads exactly those, and
places them where they belong. A key the plan drops is never read, because the
abstract tree handed to the v1 loader marks it with `PLACEHOLDER`.
"""

import os
from typing import Any

import jax
import numpy as np
from orbax.checkpoint.experimental.v1._src.loading import loading as loading_lib
from orbax.checkpoint.experimental.v1._src.metadata import loading as metadata_loading
from orbax.checkpoint.experimental.v1._src.serialization import types as serialization_types
from orbax.checkpoint.experimental.v1._src.surgery import manifest as manifest_lib
from orbax.checkpoint.experimental.v1._src.surgery import pipeline as pipeline_lib
from orbax.checkpoint.experimental.v1._src.surgery import trees

Region = manifest_lib.Region
VirtualLeaf = manifest_lib.VirtualLeaf


def _is_checkpoint_source(source: Any) -> bool:
  return isinstance(source, (str, os.PathLike))


def _replicated_sharding() -> jax.sharding.Sharding:
  devices = jax.devices()
  mesh = jax.sharding.Mesh(np.asarray(devices), ("_surgery",))
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def _sharding_of(spec: Any) -> jax.sharding.Sharding:
  sharding = getattr(spec, "sharding", None)
  if sharding is not None:
    return sharding
  return jax.sharding.SingleDeviceSharding(jax.devices()[0])


def _referenced_source_keys(manifest: manifest_lib.Manifest) -> set[str]:
  """Returns the source keys any assignment reads."""
  keys: set[str] = set()
  for leaf in manifest.values():
    for a in manifest_lib.as_virtual(leaf).assignments:
      if a.ref.source == pipeline_lib.DEFAULT_SOURCE:
        keys.add(a.ref.key)
  return keys


def read_keys(plan: pipeline_lib.Plan, source_metadata: Any) -> set[str]:
  """Returns the source keys a plan will read from a checkpoint.

  Resolves against metadata alone, so it moves no array data. Keys the plan
  drops are absent from the result and are never fetched.

  Args:
    plan: The plan to resolve.
    source_metadata: Source metadata, for example from `ocp.metadata`.

  Returns:
    The set of dotted source keys that execution will read.
  """
  resolved = plan.resolve(source_metadata)
  return _referenced_source_keys(resolved.manifest)


def _load_checkpoint_sources(
    path: Any,
    source_metadata: Any,
    meta_flat: dict[str, Any],
    needed: set[str],
) -> dict[str, Any]:
  """Loads only the source leaves a plan needs, via the v1 loader.

  Keys outside `needed` are marked with `PLACEHOLDER` in the abstract tree, so
  the loader never reads them.

  Args:
    path: The checkpoint path.
    source_metadata: The checkpoint's metadata tree.
    meta_flat: The flat view of `source_metadata`.
    needed: The dotted keys the plan will read.

  Returns:
    A flat mapping of source key to the value the loader returned.
  """
  replicated = _replicated_sharding()
  abstract_flat: dict[str, Any] = {}
  for key, leaf_meta in meta_flat.items():
    if key in needed:
      abstract_flat[key] = jax.ShapeDtypeStruct(
          tuple(leaf_meta.shape),
          leaf_meta.dtype,
          sharding=replicated,
      )
    else:
      abstract_flat[key] = loading_lib.PLACEHOLDER
  abstract_tree = trees.unflatten_like(source_metadata, abstract_flat)

  loaded = loading_lib.load(path, abstract_tree)
  return trees.flatten(loaded)


def execute(plan: pipeline_lib.Plan, sources: Any, target: Any) -> Any:
  """Runs a resolved plan and returns a tree shaped like `target`.

  Args:
    plan: The plan to execute.
    sources: A checkpoint path or an in-memory tree of source arrays.
    target: The abstract target tree: leaves are `jax.ShapeDtypeStruct` or
      arrays carrying a sharding.

  Returns:
    A tree with the structure of `target` and assembled arrays as leaves.
  """
  if _is_checkpoint_source(sources):
    source_metadata = metadata_loading.metadata(sources).metadata
    meta_flat = trees.flatten(source_metadata)
    needed = read_keys(plan, source_metadata)
    loaded_flat = _load_checkpoint_sources(
        sources, source_metadata, meta_flat, needed
    )
    # A full-key-set source keeps ops seeing the same keys a preview did; only
    # needed keys carry array data, the rest are shape-and-dtype-only leaves.
    combined: dict[str, Any] = {}
    for key, leaf_meta in meta_flat.items():
      if key in needed and not serialization_types.is_placeholder(
          loaded_flat[key]
      ):
        combined[key] = loaded_flat[key]
      else:
        combined[key] = jax.ShapeDtypeStruct(
            tuple(leaf_meta.shape), leaf_meta.dtype
        )
    resolved = plan.resolve(combined, target)
  else:
    resolved = plan.resolve(sources, target)

  resolved.report.raise_if_errors()

  target_flat = trees.flatten(target)
  result_flat: dict[str, Any] = {}
  for key, target_spec in target_flat.items():
    leaf = resolved.manifest[key]
    result_flat[key] = _assemble_leaf(
        manifest_lib.as_virtual(leaf),
        tuple(target_spec.shape),
        np.dtype(target_spec.dtype),
        _sharding_of(target_spec),
    )
  return trees.unflatten_like(target, result_flat)


def load(source: Any, plan: pipeline_lib.Plan, *, target: Any) -> Any:
  """Loads a checkpoint or tree through a plan.

  Args:
    source: A checkpoint path or an in-memory tree of source arrays.
    plan: The plan to apply.
    target: The abstract target tree.

  Returns:
    A tree with the structure of `target` and assembled arrays as leaves.
  """
  return execute(plan, source, target)


def _region_from_index(index, shape) -> Region:
  start = []
  stop = []
  for axis, sl in enumerate(index):
    lo, hi, _ = sl.indices(shape[axis])
    start.append(lo)
    stop.append(hi)
  return Region(tuple(start), tuple(stop))


def _assemble_leaf(
    leaf: VirtualLeaf,
    shape: tuple[int, ...],
    dtype: np.dtype,
    sharding: jax.sharding.Sharding,
) -> jax.Array:
  """Builds one target array by reading each shard's source regions.

  Args:
    leaf: The virtual leaf to assemble.
    shape: The global target shape.
    dtype: The target dtype.
    sharding: The target sharding.

  Returns:
    The assembled `jax.Array`.
  """

  def callback(index) -> np.ndarray:
    shard_region = _region_from_index(index, shape)
    if leaf.init is not None:
      buf = np.array(leaf.init(shard_region.shape, dtype), dtype=dtype)
    else:
      buf = np.empty(shard_region.shape, dtype=dtype)
    for a in leaf.assignments:
      hit = shard_region.intersect(a.target_region)
      if hit is None:
        continue
      source_region = manifest_lib.source_read_region(a, hit)
      chunk = a.ref.read(source_region)
      chunk = manifest_lib.align_to_target(chunk, a)
      local = tuple(
          slice(h0 - s0, h1 - s0)
          for h0, h1, s0 in zip(hit.start, hit.stop, shard_region.start)
      )
      buf[local] = chunk
    return buf

  return jax.make_array_from_callback(shape, sharding, callback)
