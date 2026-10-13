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

"""The host backend: computes box values in host memory with NumPy.

A view is laid out by striding over the fetched block, with no copy, and then
copied once into the shard buffer. A copy that transposes walks memory out of
order, so it runs in cache-sized tiles. Per-element primitives are bound on the
CPU device exactly as an op-by-op run of the function binds them.
"""

from collections.abc import Callable, Mapping
import dataclasses
import functools
from typing import Any

import jax
from jax.extend import core as jax_core
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import boxes
from orbax.checkpoint.experimental.v1._src.surgery import reads

Region = boxes.Region

# Evaluates one primitive on its operand arrays, called as `bind` would be.
ApplyFn = Callable[..., Any]

# Edge, in elements, of the square tiles a transposing copy moves at a time.
_TILE = 256

# Source and destination dtypes whose NumPy conversion rounds exactly as XLA's
# does, so a cast over a plain view can be done by the copy itself.
_EXACT_CASTS = frozenset(
    np.dtype(d) for d in (jnp.float16, jnp.bfloat16, jnp.float32)
)


@functools.cache
def _cpu_device() -> jax.Device:
  return jax.devices("cpu")[0]


def bind_on_cpu(
    primitive: jax_core.Primitive, params: dict[str, Any], *args: Any
) -> list[np.ndarray]:
  """Runs one primitive on host arrays and returns its results on host."""
  with jax.default_device(_cpu_device()):
    out = primitive.bind(*args, **params)
  return [np.asarray(o) for o in (out if primitive.multiple_results else [out])]


def strided(
    block: np.ndarray, offset: np.ndarray, lattice: np.ndarray, shape: Any
) -> np.ndarray:
  """Returns `block[offset + lattice @ l]` over `shape`, as a read-only view.

  Args:
    block: A fetched source block.
    offset: The block index of local index zero.
    lattice: The block index step per local axis.
    shape: The local shape.

  Returns:
    A strided view of `block`; no element is copied.
  """
  if block.ndim == 0:
    return np.broadcast_to(block, shape)
  first = block[tuple(slice(int(i), None) for i in offset)]
  steps = lattice.T @ np.asarray(block.strides, dtype=np.int64)
  return np.lib.stride_tricks.as_strided(
      first,
      shape=tuple(shape),
      strides=tuple(int(s) for s in steps),
      writeable=False,
  )


def view_values(
    view: boxes.View, start: np.ndarray, shape: Any, fetch: reads.FetchFn
) -> np.ndarray:
  """Fetches what a view needs over one box and strides it into place.

  Args:
    view: The view to read.
    start: The local box's first index.
    shape: The local box's shape.
    fetch: Reads source regions.

  Returns:
    The box's values as a read-only view of the fetched block.
  """
  if np.prod(shape) == 0:
    return np.empty(shape, dtype=view.ref.dtype)
  read = reads.plan_view_read(view, start, shape)
  block = np.asarray(fetch(view.ref, read.region))
  return strided(block, read.offset, read.lattice, shape)


def copy_tiled(out: np.ndarray, values: np.ndarray) -> None:
  """Copies `values` into `out`, tile by tile when the copy transposes.

  When the axis `values` is densest along is not the last axis of `out`, an
  element-by-element copy misses the cache on nearly every write. Moving
  square tiles keeps both sides of each tile in cache.

  Args:
    out: The destination; its last axis is contiguous.
    values: The source, of the same shape, laid out in any order.
  """
  last = out.ndim - 1
  if out.ndim < 2 or out.shape[last] < _TILE:
    out[...] = values
    return
  sizes = [
      abs(s) if s and n > 1 else np.inf
      for s, n in zip(values.strides, values.shape)
  ]
  inner = int(np.argmin(sizes))
  if inner == last or out.shape[inner] < _TILE:
    out[...] = values
    return
  for i in range(0, out.shape[inner], _TILE):
    for j in range(0, out.shape[last], _TILE):
      tile = [slice(None)] * out.ndim
      tile[inner] = slice(i, i + _TILE)
      tile[last] = slice(j, j + _TILE)
      out[tuple(tile)] = values[tuple(tile)]


def _exact_cast_source(expr: boxes.Expr) -> boxes.View | None:
  """Returns the view under a cast NumPy performs exactly as XLA does."""
  if (
      isinstance(expr, boxes.Apply)
      and expr.primitive.name == "convert_element_type"
      and isinstance(expr.args[0], boxes.View)
      and np.dtype(expr.args[0].ref.dtype) in _EXACT_CASTS
      and np.dtype(expr.params["new_dtype"]) in _EXACT_CASTS
  ):
    return expr.args[0]
  return None


@dataclasses.dataclass(frozen=True)
class _HostShard:
  region: Region
  buffer: np.ndarray


class HostBackend:
  """Fills each shard in a host buffer.

  Attributes:
    apply: Per-primitive overrides, keyed by primitive name. Each is called with
      the operand arrays and the equation's parameters, as `bind` would be, and
      replaces binding that primitive on the CPU.
  """

  def __init__(self, apply: Mapping[str, ApplyFn] | None = None):
    self.apply = dict(apply or {})

  def start(self, region: Region, dtype: np.dtype, device: Any) -> _HostShard:
    del device
    return _HostShard(region, np.empty(region.shape, dtype=dtype))

  def put(
      self,
      shard: _HostShard,
      box: boxes.Assignment,
      hit: Region,
      fetch: reads.FetchFn,
  ) -> None:
    """Writes one box's values for `hit` into the shard buffer.

    A plain or strided view, cast or not, is copied from the fetched block
    straight into the buffer; only per-element math builds a temporary.

    Args:
      shard: The shard being filled.
      box: The box.
      hit: The part of the box inside the shard, in target coordinates.
      fetch: Reads source regions.
    """
    out = shard.buffer[hit.shifted([-s for s in shard.region.start]).slices]
    start = box.local_start(hit)
    source = box.expr if isinstance(box.expr, boxes.View) else None
    if source is None and "convert_element_type" not in self.apply:
      source = _exact_cast_source(box.expr)
    if source is None:
      out[...] = self.evaluate(box.expr, start, hit.shape, fetch)
      return
    # Overflow to infinity is the conversion's defined result, as in XLA.
    with np.errstate(over="ignore"):
      copy_tiled(out, view_values(source, start, hit.shape, fetch))

  def evaluate(
      self,
      expr: boxes.Expr,
      start: np.ndarray,
      shape: Any,
      fetch: reads.FetchFn,
  ) -> np.ndarray:
    """Returns `expr` over the local box at `start` with `shape`."""
    if isinstance(expr, boxes.View):
      return view_values(expr, start, shape, fetch)
    if isinstance(expr, boxes.Const):
      return np.broadcast_to(expr.value, shape)
    args = [self.evaluate(a, start, shape, fetch) for a in expr.args]
    override = self.apply.get(expr.primitive.name)
    if override is not None:
      return np.asarray(override(*args, **expr.params))
    return bind_on_cpu(expr.primitive, expr.params, *args)[0]

  def finish(self, shard: _HostShard) -> np.ndarray:
    return shard.buffer
