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


"""Loads a source through a surgery function as a sharded read.

Each device shard of each target array asks which source boxes land inside it,
reads exactly the bytes those boxes need, and is sent to its devices as soon as
it is filled. Where the values are laid out is the backend's choice: in host
memory for CPU targets, on the device for accelerators.
"""

from collections.abc import Callable
import time
from typing import Any

import jax
import numpy as np
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.v1._src.surgery import boxes
from orbax.checkpoint.experimental.v1._src.surgery import fill
from orbax.checkpoint.experimental.v1._src.surgery import reads
from orbax.checkpoint.experimental.v1._src.surgery import rules as rules_lib
from orbax.checkpoint.experimental.v1._src.surgery import sources
from orbax.checkpoint.experimental.v1._src.surgery import tracing

Region = boxes.Region


def _region_from_index(index: tuple[slice, ...], shape: Any) -> Region:
  bounds = [sl.indices(n)[:2] for sl, n in zip(index, shape)]
  return Region(tuple(lo for lo, _ in bounds), tuple(hi for _, hi in bounds))


def _assemble(
    leaf: boxes.VirtualLeaf,
    spec: Any,
    backend_for: Callable[[jax.Device], fill.Backend],
    stats: reads.LoadStats | None,
) -> jax.Array:
  """Fills each distinct shard of one target once and sends it to its devices.

  A shard held by several devices, as replicas are, is built once, on the
  first of them, and copied to the rest.

  Args:
    leaf: The boxes of the target.
    spec: The abstract target leaf, carrying its sharding if it has one.
    backend_for: Chooses the backend for a target device.
    stats: Collects read and host timings, if given.

  Returns:
    The assembled `jax.Array`.
  """
  shape = tuple(spec.shape)
  sharding = getattr(
      spec, "sharding", None
  ) or jax.sharding.SingleDeviceSharding(jax.devices()[0])
  index_map = sharding.addressable_devices_indices_map(shape)
  devices: dict[Region, list[jax.Device]] = {}
  for device, index in index_map.items():
    devices.setdefault(_region_from_index(index, shape), []).append(device)
  per_device: dict[jax.Device, jax.Array] = {}

  def send(region: Region, shard: np.ndarray | jax.Array) -> None:
    for device in devices[region]:
      per_device[device] = jax.device_put(shard, device)

  fill.fill(
      leaf,
      {region: held[0] for region, held in devices.items()},
      send,
      backend_for=backend_for,
      stats=stats,
  )
  return jax.make_array_from_single_device_arrays(
      shape, sharding, [per_device[d] for d in index_map]
  )


def load(
    source: Any,
    fn: Callable[[Any], Any],
    *,
    target: Any,
    rules: rules_lib.Rules = rules_lib.DEFAULT_RULES,
    backend: fill.Backend | None = None,
    stats: reads.LoadStats | None = None,
) -> Any:
  """Loads `fn(source)` straight into sharded target arrays.

  `fn` receives the source tree as stored, flattened to dotted keys for a
  checkpoint, and returns the target tree using ordinary JAX operations. It is
  traced on shapes alone; then each shard reads only the bytes that land in
  it.

  Args:
    source: A checkpoint path, a mapping from name to checkpoint path, or an
      in-memory tree of arrays.
    fn: The surgery function.
    target: The abstract target tree: leaves are `jax.ShapeDtypeStruct` or
      arrays carrying a sharding.
    rules: How each primitive becomes boxes; `DEFAULT_RULES` unless a caller
      adds, replaces or removes rules.
    backend: Where box values are computed. `None` builds shards on the host
      for CPU targets and on the device for accelerator targets.
    stats: Filled with read, host and transfer timings, if given.

  Returns:
    A tree with the structure of `target` and assembled arrays as leaves.

  Raises:
    SurgeryError: If `fn` does not produce exactly the target's keys, shapes
      and dtypes.
  """
  start = time.perf_counter()
  if sources.is_path(source) or sources.is_path_mapping(source):
    source = sources.open_checkpoints(source, fn, rules)
  resolved = tracing.resolve(fn, source, target, rules=rules)
  resolved.report.raise_if_errors()
  backend_for = fill.default_backend if backend is None else (lambda _: backend)
  result = {
      key: _assemble(resolved.manifest[key], spec, backend_for, stats)
      for key, spec in tree_utils.to_flat_dict(target, sep=".").items()
  }
  if stats is not None:
    waiting = time.perf_counter()
    jax.block_until_ready(list(result.values()))
    done = time.perf_counter()
    stats.transfer_seconds += done - waiting
    stats.total_seconds += done - start
  return tree_utils.from_flat_dict(result, target=target, sep=".")
