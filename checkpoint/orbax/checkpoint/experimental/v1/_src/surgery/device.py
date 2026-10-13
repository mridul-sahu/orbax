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

"""The device backend: computes box values on the target device with XLA.

The host only reads bytes. Each fetched block is sent to the shard's device in
the order storage returned it, and any reordering a view needs (transpose,
reverse, strided interleave, broadcast) runs there as XLA slice, reshape, rev,
transpose and broadcast operations, where memory bandwidth is far higher than
on the host. A block that already sits in the box's order is sent as is.
Per-element primitives are bound eagerly on the device, one at a time, so they
round exactly as an op-by-op run of the function does. Boxes are then placed
into the shard by one compiled update, cached by the shard's layout, so a
layer repeated many times compiles once.
"""

from collections.abc import Mapping
import dataclasses
import functools
import math
import threading
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import boxes
from orbax.checkpoint.experimental.v1._src.surgery import host
from orbax.checkpoint.experimental.v1._src.surgery import reads

Region = boxes.Region
Shape = boxes.Shape


def _relayout(
    block: jax.Array,
    offset: tuple[int, ...],
    lattice: tuple[tuple[int, ...], ...],
    shape: Shape,
) -> jax.Array:
  """Returns `block[offset + lattice @ l]` over `shape`, with XLA operations.

  Each block axis is taken by one strided slice and split into the box axes
  that walk it, outermost first. That works when those axes nest, each step
  being the next one's step times its extent, which is what transposes,
  reshapes, flips and strided slices produce. Anything else is a gather.

  Args:
    block: The fetched block, on the device.
    offset: The block index of local index zero.
    lattice: The block index step per local axis, one row per block axis.
    shape: The local shape.

  Returns:
    The box's values.
  """
  groups: list[list[tuple[int, int]]] = [[] for _ in offset]
  for j, extent in enumerate(shape):
    rows = [i for i, row in enumerate(lattice) if row[j] and extent > 1]
    if len(rows) > 1:
      return _gather(block, offset, lattice, shape)
    if rows:
      groups[rows[0]].append((j, lattice[rows[0]][j]))
  starts, limits, strides, order, flips = [], [], [], [], []
  for i, group in enumerate(groups):
    group.sort(key=lambda g: -abs(g[1]))
    if any(
        abs(c) != abs(d) * shape[k] for (_, c), (k, d) in zip(group, group[1:])
    ):
      return _gather(block, offset, lattice, shape)
    step = abs(group[-1][1]) if group else 1
    count = math.prod(shape[j] for j, _ in group)
    lo = offset[i] + sum(min(0, c * (shape[j] - 1)) for j, c in group)
    starts.append(lo)
    limits.append(lo + (count - 1) * step + 1)
    strides.append(step)
    for j, c in group:
      if c < 0:
        flips.append(len(order))
      order.append(j)
  x = lax.slice(block, starts, limits, strides)
  x = x.reshape([shape[j] for j in order])
  if flips:
    x = lax.rev(x, flips)
  x = jnp.transpose(x, np.argsort(order))
  return lax.broadcast_in_dim(x, shape, sorted(order))


def _gather(
    block: jax.Array,
    offset: tuple[int, ...],
    lattice: tuple[tuple[int, ...], ...],
    shape: Shape,
) -> jax.Array:
  local = np.indices(shape, dtype=np.int64)
  index = tuple(
      o + np.tensordot(np.asarray(row), local, axes=1)
      for o, row in zip(offset, lattice)
  )
  return block[index]


@functools.lru_cache(maxsize=None)
def _relayout_fn(
    offset: tuple[int, ...],
    lattice: tuple[tuple[int, ...], ...],
    shape: Shape,
) -> Any:
  return jax.jit(
      functools.partial(_relayout, offset=offset, lattice=lattice, shape=shape)
  )


@functools.lru_cache(maxsize=None)
def _assemble_fn(
    shape: Shape, dtype: np.dtype, starts: tuple[tuple[int, ...], ...]
) -> Any:
  def assemble(*parts):
    out = jnp.zeros(shape, dtype)
    for start, part in zip(starts, parts):
      out = lax.dynamic_update_slice(out, part.astype(dtype), start)
    return out

  return jax.jit(assemble)


@dataclasses.dataclass
class _DeviceShard:
  region: Region
  dtype: np.dtype
  device: jax.Device
  parts: list[tuple[tuple[int, ...], jax.Array]] = dataclasses.field(
      default_factory=list
  )
  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


class DeviceBackend:
  """Builds each shard on its target device.

  Attributes:
    apply: Per-primitive overrides, keyed by primitive name. Each is called with
      the operand arrays, on the device, and the equation's parameters, as
      `bind` would be.
  """

  def __init__(self, apply: Mapping[str, host.ApplyFn] | None = None):
    self.apply = dict(apply or {})

  def start(
      self, region: Region, dtype: np.dtype, device: jax.Device
  ) -> _DeviceShard:
    return _DeviceShard(region, np.dtype(dtype), device)

  def put(
      self,
      shard: _DeviceShard,
      box: boxes.Assignment,
      hit: Region,
      fetch: reads.FetchFn,
  ) -> None:
    """Computes one box over `hit` on the shard's device."""
    value = self.evaluate(
        box.expr, box.local_start(hit), hit.shape, fetch, shard.device
    )
    local = tuple(h - s for h, s in zip(hit.start, shard.region.start))
    with shard.lock:
      shard.parts.append((local, value))

  def evaluate(
      self,
      expr: boxes.Expr,
      start: np.ndarray,
      shape: Shape,
      fetch: reads.FetchFn,
      device: jax.Device,
  ) -> jax.Array:
    """Returns `expr` over the local box at `start`, on `device`."""
    if isinstance(expr, boxes.Const):
      return jnp.full(shape, expr.value, dtype=expr.value.dtype, device=device)
    if isinstance(expr, boxes.View):
      return self._view(expr, start, shape, fetch, device)
    args = [self.evaluate(a, start, shape, fetch, device) for a in expr.args]
    override = self.apply.get(expr.primitive.name)
    with jax.default_device(device):
      if override is not None:
        return override(*args, **expr.params)
      return expr.primitive.bind(*args, **expr.params)

  def _view(
      self,
      view: boxes.View,
      start: np.ndarray,
      shape: Shape,
      fetch: reads.FetchFn,
      device: jax.Device,
  ) -> jax.Array:
    """Sends a view's block to `device` and lays it out there if needed."""
    if math.prod(shape) == 0:
      return jnp.zeros(shape, view.ref.dtype, device=device)
    read = reads.plan_view_read(view, start, shape)
    block = np.asarray(fetch(view.ref, read.region))
    in_order = host.strided(block, read.offset, read.lattice, shape)
    if in_order.flags.c_contiguous:
      return jax.device_put(in_order, device)
    relayout = _relayout_fn(
        tuple(int(o) for o in read.offset),
        tuple(tuple(int(c) for c in row) for row in read.lattice),
        tuple(shape),
    )
    return relayout(jax.device_put(block, device))

  def finish(self, shard: _DeviceShard) -> jax.Array:
    """Places every computed box into one array on the shard's device."""
    parts = sorted(shard.parts, key=lambda p: p[0])
    if len(parts) == 1 and parts[0][1].shape == shard.region.shape:
      return parts[0][1].astype(shard.dtype)
    assemble = _assemble_fn(
        tuple(shard.region.shape),
        shard.dtype,
        tuple(start for start, _ in parts),
    )
    with jax.default_device(shard.device):
      return assemble(*(value for _, value in parts))
