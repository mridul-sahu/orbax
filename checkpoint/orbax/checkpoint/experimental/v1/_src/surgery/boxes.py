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

"""The box model a surgery function resolves to.

A manifest maps each target key to a `VirtualLeaf`: a set of disjoint boxes,
each carrying an expression whose leaves are integer affine views of source
arrays. Layout changes rewrite the views, never the data, so a chain of them of
any length leaves every box a single hop from stored bytes to its place in the
target. This module holds only the description; `reads` plans the bytes a box
needs and the backends in `host` and `device` compute its values.
"""

from collections.abc import Callable, Iterator, Sequence
import dataclasses
import math
from typing import Any

from jax.extend import core as jax_core
import numpy as np

Shape = tuple[int, ...]

# Reads a rectangular sub-range of a source array and returns it as a host
# ndarray. `None` on a metadata-only manifest, where no data is available.
ReadFn = Callable[["Region"], np.ndarray]

# The namespace of sources computed during a load rather than stored.
DERIVED_SOURCE = "__derived__"


@dataclasses.dataclass(frozen=True)
class Region:
  """A rectangular selection of an array: a start and stop per axis.

  Plan geometry is always dense. A read request may carry a `step`, selecting
  every `step`-th index from `start` up to `stop`, so a strided view fetches
  only the elements it uses; readers honor it through `slices`.

  Attributes:
    start: Inclusive lower bound per axis.
    stop: Exclusive upper bound per axis.
    step: Stride per axis for a read request, or `None` for a dense region.
  """

  start: tuple[int, ...]
  stop: tuple[int, ...]
  step: tuple[int, ...] | None = None

  @property
  def ndim(self) -> int:
    return len(self.start)

  @property
  def steps(self) -> tuple[int, ...]:
    return self.step if self.step is not None else (1,) * self.ndim

  @property
  def shape(self) -> Shape:
    return tuple(
        max(0, -((lo - hi) // s))
        for lo, hi, s in zip(self.start, self.stop, self.steps)
    )

  @property
  def size(self) -> int:
    return math.prod(self.shape)

  @property
  def slices(self) -> tuple[slice, ...]:
    return tuple(
        slice(lo, hi, s if self.step is not None else None)
        for lo, hi, s in zip(self.start, self.stop, self.steps)
    )

  @classmethod
  def full(cls, shape: Shape) -> "Region":
    """Returns the region spanning the whole array of the given shape."""
    return cls(tuple(0 for _ in shape), tuple(shape))

  def intersect(self, other: "Region") -> "Region | None":
    """Returns the overlap with `other`, or `None` if they are disjoint.

    Args:
      other: A region of the same rank, in the same coordinate frame.

    Returns:
      The intersection region, or `None` when the overlap is empty.
    """
    start = tuple(max(a, b) for a, b in zip(self.start, other.start))
    stop = tuple(min(a, b) for a, b in zip(self.stop, other.stop))
    if any(hi <= lo for lo, hi in zip(start, stop)):
      return None
    return Region(start, stop)

  def shifted(self, offsets: Sequence[int]) -> "Region":
    """Returns this dense region moved by `offsets`, one per axis."""
    return Region(
        tuple(a + o for a, o in zip(self.start, offsets)),
        tuple(b + o for b, o in zip(self.stop, offsets)),
    )


@dataclasses.dataclass(frozen=True)
class LeafRef:
  """A readable array described without reading it.

  Attributes:
    source: The namespace the key lives in.
    key: The key within that source.
    shape: The array shape.
    dtype: The array dtype.
    read: Reads any sub-range on demand, or `None` on a metadata-only manifest.
  """

  source: str
  key: str
  shape: Shape
  dtype: np.dtype
  read: ReadFn | None = None


@dataclasses.dataclass(frozen=True)
class DerivedRef(LeafRef):
  """A source whose values are computed from other sources.

  Reading any region of it reads its `inputs` in full, so a plan that routes a
  leaf through one is reported as materialized. Execution opens the stores of
  `inputs`, not of the derived ref itself.

  Attributes:
    inputs: The sources the computation reads.
  """

  inputs: tuple[LeafRef, ...] = ()


@dataclasses.dataclass(frozen=True)
class StoredArray:
  """A source array on storage whose sub-ranges are read on demand.

  Attributes:
    shape: The array shape.
    dtype: The array dtype.
    read: Reads any sub-range from storage.
  """

  shape: Shape
  dtype: np.dtype
  read: ReadFn


@dataclasses.dataclass(frozen=True, eq=False)
class View:
  """Source values at `origin + matrix @ local` for each local index.

  Attributes:
    ref: The source array.
    origin: The source index of local index zero, one entry per source axis.
    matrix: Integer map from local to source coordinates, shaped (source rank,
      local rank). A zero column broadcasts; a negative entry reverses.
  """

  ref: LeafRef
  origin: np.ndarray
  matrix: np.ndarray


@dataclasses.dataclass(frozen=True, eq=False)
class Const:
  """A scalar repeated over the whole box.

  Attributes:
    value: A zero-dimensional array carrying the value and its dtype.
  """

  value: np.ndarray


@dataclasses.dataclass(frozen=True, eq=False)
class Apply:
  """An elementwise primitive over expressions of the same box.

  Attributes:
    primitive: The JAX primitive, bound exactly as a traced function binds it.
    params: The primitive's parameters from the jaxpr equation.
    args: The operand expressions.
  """

  primitive: jax_core.Primitive
  params: dict[str, Any]
  args: tuple["Expr", ...]


Expr = View | Const | Apply


def views(expr: Expr) -> Iterator[View]:
  """Yields every view an expression reads."""
  if isinstance(expr, View):
    yield expr
  elif isinstance(expr, Apply):
    for arg in expr.args:
      yield from views(arg)


def substitute(expr: Expr, offset: np.ndarray, linear: np.ndarray) -> Expr:
  """Re-expresses `expr` in new local coordinates `o`, where old `l = c + T o`.

  Args:
    expr: The expression in the old local coordinates.
    offset: `c`, one entry per old local axis.
    linear: `T`, shaped (old local rank, new local rank).

  Returns:
    The same values, indexed by the new local coordinates.
  """
  if isinstance(expr, View):
    return View(
        expr.ref, expr.origin + expr.matrix @ offset, expr.matrix @ linear
    )
  if isinstance(expr, Apply):
    return Apply(
        expr.primitive,
        expr.params,
        tuple(substitute(a, offset, linear) for a in expr.args),
    )
  return expr


@dataclasses.dataclass(frozen=True, eq=False)
class Assignment:
  """One box of a virtual leaf and the expression that fills it.

  Attributes:
    target_region: The box, in target coordinates.
    expr: The expression over the box's local coordinates.
  """

  target_region: Region
  expr: Expr

  def local_start(self, hit: Region) -> np.ndarray:
    """Returns where `hit`, a sub-region of the box, starts in its frame."""
    return np.asarray(hit.start, dtype=np.int64) - np.asarray(
        self.target_region.start, dtype=np.int64
    )

  def refs(self) -> tuple[LeafRef, ...]:
    return tuple(v.ref for v in views(self.expr))


@dataclasses.dataclass(frozen=True)
class VirtualLeaf:
  """A target array under construction.

  Attributes:
    shape: The shape of the assembled array.
    dtype: The dtype of the assembled array.
    assignments: The disjoint boxes that fill it, in no particular order.
  """

  shape: Shape
  dtype: np.dtype
  assignments: tuple[Assignment, ...]


Manifest = dict[str, VirtualLeaf]


def as_virtual(ref: LeafRef) -> VirtualLeaf:
  """Returns a leaf holding all of `ref` as one identity box."""
  n = len(ref.shape)
  view = View(ref, np.zeros(n, np.int64), np.eye(n, dtype=np.int64))
  return VirtualLeaf(
      tuple(ref.shape),
      np.dtype(ref.dtype),
      (Assignment(Region.full(tuple(ref.shape)), view),),
  )


def stored_refs(leaf: VirtualLeaf) -> list[LeafRef]:
  """Returns the stored sources a leaf reads, expanding derived sources."""
  pending = [ref for a in leaf.assignments for ref in a.refs()]
  seen: dict[int, LeafRef] = {}
  while pending:
    ref = pending.pop()
    if id(ref) in seen:
      continue
    seen[id(ref)] = ref
    if isinstance(ref, DerivedRef):
      pending.extend(ref.inputs)
  return [ref for ref in seen.values() if not isinstance(ref, DerivedRef)]
