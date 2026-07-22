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

"""The intermediate representation for plan-based model surgery.

A manifest is a flat mapping from key to leaf, where each leaf is either a bare
`LeafRef` (untouched so far) or a `VirtualLeaf` produced by an operation. The
types here describe an array without reading it: an operation moves regions, not
data, and regions collapse under composition, so a chain of operations of any
length leaves every assignment a single hop from stored bytes to final
placement.
"""

from collections.abc import Callable
import dataclasses
import math

import numpy as np

Shape = tuple[int, ...]

# Reads a rectangular sub-range of a source array and returns it as a host
# ndarray. `None` on a metadata-only manifest, where no data is available.
ReadFn = Callable[["Region"], np.ndarray]

# Fills a region that no assignment covers, given its shape and dtype.
InitFn = Callable[[Shape, np.dtype], np.ndarray]


@dataclasses.dataclass(frozen=True)
class Region:
  """A rectangular selection of an array: a start and stop per axis.

  Attributes:
    start: Inclusive lower bound per axis.
    stop: Exclusive upper bound per axis.
  """

  start: tuple[int, ...]
  stop: tuple[int, ...]

  @property
  def ndim(self) -> int:
    return len(self.start)

  @property
  def shape(self) -> Shape:
    return tuple(hi - lo for lo, hi in zip(self.start, self.stop))

  @property
  def size(self) -> int:
    return math.prod(self.shape)

  @property
  def slices(self) -> tuple[slice, ...]:
    return tuple(slice(lo, hi) for lo, hi in zip(self.start, self.stop))

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

  def compose(self, inner: "Region") -> "Region":
    """Maps a sub-region expressed in this region's local frame to absolute
    coordinates.

    Args:
      inner: A region whose coordinates are relative to this region's start.

    Returns:
      The absolute region, obtained by offsetting `inner` by this start.
    """
    return Region(
        tuple(s + i for s, i in zip(self.start, inner.start)),
        tuple(s + i for s, i in zip(self.start, inner.stop)),
    )

  def with_inserted_axis(self, axis: int, index: int) -> "Region":
    """Returns a region with a unit-width axis inserted at `axis`.

    Args:
      axis: Position of the new axis.
      index: The single index the new axis selects.

    Returns:
      A region of rank one greater than this one.
    """
    start = self.start[:axis] + (index,) + self.start[axis:]
    stop = self.stop[:axis] + (index + 1,) + self.stop[axis:]
    return Region(start, stop)

  def without_axis(self, axis: int) -> "Region":
    """Returns a region with `axis` removed.

    Args:
      axis: The axis to drop.

    Returns:
      A region of rank one less than this one.
    """
    return Region(
        self.start[:axis] + self.start[axis + 1 :],
        self.stop[:axis] + self.stop[axis + 1 :],
    )


@dataclasses.dataclass(frozen=True)
class LeafRef:
  """A readable array described without reading it.

  Attributes:
    source: The namespace the key lives in (for example "vision" in a merge).
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
class Assignment:
  """Maps a region of one source onto a region of a target.

  `inserted_axis` and `sliced_source_axis` record the single rank change some
  operations introduce: a stack inserts a target axis the source lacks, an
  unstack reads a unit-width source axis the target lacks. At most one is set.

  Attributes:
    target_region: Where in the virtual leaf the bytes land.
    ref: The source the bytes come from.
    source_region: Which part of the source supplies them.
    cast: Convert to this dtype during streaming, or `None` to keep the source
      dtype.
    inserted_axis: A target axis with no source counterpart, or `None`.
    sliced_source_axis: A unit-width source axis with no target counterpart, or
      `None`.
  """

  target_region: Region
  ref: LeafRef
  source_region: Region
  cast: np.dtype | None = None
  inserted_axis: int | None = None
  sliced_source_axis: int | None = None


@dataclasses.dataclass(frozen=True)
class VirtualLeaf:
  """A target array under construction.

  Attributes:
    shape: The shape of the assembled array.
    dtype: The dtype of the assembled array.
    assignments: The source regions that fill it, in no particular order.
    init: Fills regions no assignment covers, or `None` if full coverage is
      required.
    transform: A whole-array function applied after assembly, or `None`. Its
      presence forces a full materialization, so it is the one escape from
      per-shard streaming.
  """

  shape: Shape
  dtype: np.dtype
  assignments: tuple[Assignment, ...]
  init: InitFn | None = None
  transform: Callable[[np.ndarray], np.ndarray] | None = None


Leaf = LeafRef | VirtualLeaf
Manifest = dict[str, Leaf]


@dataclasses.dataclass(frozen=True)
class StoredArray:
  """A source array on storage whose sub-ranges are read on demand.

  Used as a source spec so a checkpoint-backed leaf reads only the regions a
  plan places, never the whole array.

  Attributes:
    shape: The array shape.
    dtype: The array dtype.
    read: Reads any sub-range from storage.
  """

  shape: Shape
  dtype: np.dtype
  read: ReadFn


def as_virtual(leaf: Leaf) -> VirtualLeaf:
  """Views any leaf as a `VirtualLeaf`.

  A bare `LeafRef` becomes a virtual leaf with a single identity assignment, so
  the executor consumes every leaf the same way.

  Args:
    leaf: A `LeafRef` or `VirtualLeaf`.

  Returns:
    The leaf as a `VirtualLeaf`.
  """
  if isinstance(leaf, VirtualLeaf):
    return leaf
  full = Region.full(leaf.shape)
  return VirtualLeaf(
      shape=leaf.shape,
      dtype=leaf.dtype,
      assignments=(
          Assignment(target_region=full, ref=leaf, source_region=full),
      ),
  )


def source_read_region(assignment: Assignment, hit: Region) -> Region:
  """Maps a target-space overlap to the source region that supplies it.

  Args:
    assignment: The assignment whose source is being read.
    hit: The overlap between a target shard and `assignment.target_region`, in
      target coordinates.

  Returns:
    The region to read from the source array, in source coordinates.
  """
  tr = assignment.target_region
  local = Region(
      tuple(h - t for h, t in zip(hit.start, tr.start)),
      tuple(h - t for h, t in zip(hit.stop, tr.start)),
  )
  if assignment.inserted_axis is not None:
    local = local.without_axis(assignment.inserted_axis)
  elif assignment.sliced_source_axis is not None:
    local = local.with_inserted_axis(assignment.sliced_source_axis, 0)
  return assignment.source_region.compose(local)


def align_to_target(array: np.ndarray, assignment: Assignment) -> np.ndarray:
  """Reshapes a source read to match its target overlap.

  Args:
    array: The array read from the source.
    assignment: The assignment the read belongs to.

  Returns:
    The array with the assignment's rank change applied.
  """
  if assignment.inserted_axis is not None:
    return np.expand_dims(array, assignment.inserted_axis)
  if assignment.sliced_source_axis is not None:
    return np.squeeze(array, assignment.sliced_source_axis)
  return array


def assemble_host(leaf: VirtualLeaf) -> np.ndarray:
  """Assembles a virtual leaf fully on host, applying any transform.

  Used where the whole array is needed at once: a `compute` transform, or a
  compute leaf feeding a shard.

  Args:
    leaf: The virtual leaf to assemble.

  Returns:
    The assembled host array.
  """
  if leaf.init is not None:
    buf = np.array(leaf.init(leaf.shape, leaf.dtype), dtype=leaf.dtype)
  else:
    buf = np.empty(leaf.shape, dtype=leaf.dtype)
  full = Region.full(leaf.shape)
  for a in leaf.assignments:
    hit = full.intersect(a.target_region)
    if hit is None:
      continue
    chunk = align_to_target(a.ref.read(source_read_region(a, hit)), a)
    if a.cast is not None:
      chunk = chunk.astype(a.cast)
    buf[hit.slices] = chunk
  if leaf.transform is not None:
    buf = np.asarray(leaf.transform(buf)).astype(leaf.dtype, copy=False)
  return buf


def overlapping_assignments(
    assignments: tuple[Assignment, ...],
) -> tuple[int, int] | None:
  """Returns the first pair of assignments whose target regions overlap.

  Args:
    assignments: The assignments of a single virtual leaf.

  Returns:
    The indices of an overlapping pair, or `None` when all are disjoint.
  """
  for i in range(len(assignments)):
    for j in range(i + 1, len(assignments)):
      if (
          assignments[i].target_region.intersect(assignments[j].target_region)
          is not None
      ):
        return (i, j)
  return None


def covered_size(assignments: tuple[Assignment, ...]) -> int:
  """Returns the total volume the assignments claim.

  This equals the union volume only when the assignments are disjoint, which
  the caller verifies separately.

  Args:
    assignments: The assignments of a single virtual leaf.

  Returns:
    The summed volume of every assignment's target region.
  """
  return sum(a.target_region.size for a in assignments)
