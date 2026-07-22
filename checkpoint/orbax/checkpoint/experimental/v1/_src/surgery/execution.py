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
places them where they belong. A checkpoint-backed source opens a TensorStore
per needed leaf and reads only the requested sub-ranges; a key the plan drops is
never opened, so its bytes are never read.
"""

import os
from typing import Any

from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint.experimental.v1._src.metadata import loading as metadata_loading
from orbax.checkpoint.experimental.v1._src.surgery import manifest as manifest_lib
from orbax.checkpoint.experimental.v1._src.surgery import pipeline as pipeline_lib
from orbax.checkpoint.experimental.v1._src.surgery import trees
import tensorstore as ts

Region = manifest_lib.Region
VirtualLeaf = manifest_lib.VirtualLeaf

# Standard v1 pytree checkpointable subdirectory names, in resolution order.
_CHECKPOINTABLE_NAMES = ("pytree", "state")


def _is_checkpoint_source(source: Any) -> bool:
  return isinstance(source, (str, os.PathLike))


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


def _resolve_leaf_directory(path: Any) -> tuple[str, bool]:
  """Finds the array directory of a pytree checkpoint and its OCDBT flag.

  The arrays live in the checkpointable subdirectory that holds a `_METADATA`
  file. OCDBT is inferred from the presence of a `manifest.ocdbt` file there.

  Args:
    path: The checkpoint path.

  Returns:
    The array directory as a posix string, and whether it is an OCDBT store.
  """
  base = epath.Path(path)
  if (base / "_METADATA").exists():
    leaf_dir = base
  else:
    candidates = [
        d for d in base.iterdir() if (d / "_METADATA").exists()
    ]
    if not candidates:
      raise ValueError(f"No pytree checkpointable found under {path}.")
    by_name = {d.name: d for d in candidates}
    leaf_dir = next(
        (by_name[n] for n in _CHECKPOINTABLE_NAMES if n in by_name),
        sorted(candidates, key=lambda d: d.name)[0],
    )
  use_ocdbt = (leaf_dir / "manifest.ocdbt").exists()
  return leaf_dir.as_posix(), use_ocdbt


def _open_stored_reader(
    leaf_dir: str, name: str, use_ocdbt: bool
) -> manifest_lib.ReadFn:
  """Opens a TensorStore for one leaf and returns a sub-range reader.

  The zarr version is not recorded separately, so v3 is tried first and v2 is
  the fallback. Opening reads only the array metadata, not its data.

  Args:
    leaf_dir: The array directory.
    name: The on-disk parameter name (the dotted key).
    use_ocdbt: Whether the store is OCDBT.

  Returns:
    A function reading a region of the leaf from storage.
  """
  last_error: Exception | None = None
  for use_zarr3 in (True, False):
    spec = ts_utils.ArrayReadSpec(
        leaf_dir, name, use_zarr3, use_ocdbt=use_ocdbt
    ).json
    try:
      store = ts.open(
          ts.Spec(spec),
          open=True,
          context=ts_utils.get_ts_context(use_ocdbt=use_ocdbt),
      ).result()
    except ValueError as e:
      last_error = e
      continue

    def read(region: Region, store=store) -> np.ndarray:
      return np.asarray(store[region.slices].read().result())

    return read
  raise ValueError(f"Could not open leaf {name!r} in {leaf_dir}: {last_error}")


def _checkpoint_sources(
    path: Any, plan: pipeline_lib.Plan
) -> dict[str, Any]:
  """Builds a flat source spec for a checkpoint.

  Needed leaves become `StoredArray`s that read sub-ranges from storage; the
  rest become shape-and-dtype-only leaves that are never opened, so dropped
  keys are never read.

  Args:
    path: The checkpoint path.
    plan: The plan being executed.

  Returns:
    A flat mapping of source key to spec.
  """
  source_metadata = metadata_loading.metadata(path).metadata
  meta_flat = trees.flatten(source_metadata)
  needed = read_keys(plan, source_metadata)
  leaf_dir, use_ocdbt = _resolve_leaf_directory(path)

  combined: dict[str, Any] = {}
  for key, leaf_meta in meta_flat.items():
    if key in needed:
      combined[key] = manifest_lib.StoredArray(
          shape=tuple(leaf_meta.shape),
          dtype=np.dtype(leaf_meta.dtype),
          read=_open_stored_reader(leaf_dir, key, use_ocdbt),
      )
    else:
      combined[key] = jax.ShapeDtypeStruct(
          tuple(leaf_meta.shape), leaf_meta.dtype
      )
  return combined


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
    resolved = plan.resolve(_checkpoint_sources(sources, plan), target)
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
