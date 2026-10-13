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


"""Read planning: which source bytes a box needs, and how reads are merged.

A view over a box lands, along each source axis, on indices spaced by the gcd
of that axis's steps, so its read selects just that lattice. The reads every
box of one shard needs are then merged per source when the merged rectangle
wastes little, and each box takes its values as a slice of a merged block.
Both backends plan reads here, so a host fill and a device fill read the same
bytes.
"""

from collections.abc import Callable, Iterable, Sequence
import dataclasses
import math
import threading
import time
from typing import Any

import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import boxes

Region = boxes.Region
LeafRef = boxes.LeafRef
Shape = boxes.Shape

# Reads one region of one source; a `ReadBatch` serves these from merged reads.
FetchFn = Callable[[LeafRef, Region], np.ndarray]


def direct_fetch(ref: LeafRef, region: Region) -> np.ndarray:
  """Reads `region` straight from `ref`, with no batching."""
  return ref.read(region)


@dataclasses.dataclass(frozen=True, eq=False)
class ViewRead:
  """The source read one view needs over one box, and how to index it.

  Attributes:
    region: The source region to fetch, with a step where the lattice is
      sparse.
    offset: The index, in the fetched block, of the box's first element.
    lattice: How far the block index moves per step along each box axis,
      shaped (source rank, box rank).
  """

  region: Region
  offset: np.ndarray
  lattice: np.ndarray


def plan_view_read(
    view: boxes.View, start: np.ndarray, shape: Shape
) -> ViewRead:
  """Plans the source read for a view over the local box at `start`.

  Along each source axis the view only lands on indices spaced by the gcd of
  that axis's steps, so the read selects just that lattice: taking every other
  row reads half the rows, not their bounding rectangle.

  Args:
    view: The view to read.
    start: The local box's first index.
    shape: The local box's shape; must not be empty.

  Returns:
    The read and how the box indexes the fetched block.

  Raises:
    ValueError: If the view reaches outside its source.
  """
  base = view.origin + view.matrix @ start
  extent = view.matrix * (np.asarray(shape, dtype=np.int64) - 1)
  lo = base + np.minimum(extent, 0).sum(axis=1)
  hi = base + np.maximum(extent, 0).sum(axis=1) + 1
  if np.any(lo < 0) or np.any(hi > np.asarray(view.ref.shape, dtype=np.int64)):
    raise ValueError(
        f"view of {view.ref.key!r} reaches outside its source: [{lo}, {hi})"
        f" of {view.ref.shape}"
    )
  moving = np.abs(view.matrix[:, np.asarray(shape) > 1])
  step = np.asarray(
      [math.gcd(*(int(c) for c in row)) or 1 for row in moving],
      dtype=np.int64,
  )
  region = Region(
      tuple(int(x) for x in lo),
      tuple(int(x) for x in hi),
      tuple(int(x) for x in step) if np.any(step > 1) else None,
  )
  return ViewRead(region, (base - lo) // step, view.matrix // step[:, None])


def read_requests(
    box: boxes.Assignment, hit: Region
) -> list[tuple[LeafRef, Region]]:
  """Returns the source regions filling `hit`, a sub-region of `box`, reads."""
  if hit.size == 0:
    return []
  start = box.local_start(hit)
  return [
      (view.ref, plan_view_read(view, start, hit.shape).region)
      for view in boxes.views(box.expr)
  ]


# Two reads of one source merge into one when the merged rectangle wastes at
# most this fraction of the bytes the reads need, or at most this many bytes.
# A ranged read has a fixed cost, so a small amount of waste pays for itself.
MERGE_SLACK_FRACTION = 0.25
MERGE_SLACK_BYTES = 1 << 20


def _bounding(a: Region, b: Region) -> Region | None:
  """Returns the smallest region holding both, if their lattices agree."""
  if a.ndim != b.ndim or a.steps != b.steps:
    return None
  if any((x - y) % s for x, y, s in zip(a.start, b.start, a.steps)):
    return None
  return Region(
      tuple(min(x, y) for x, y in zip(a.start, b.start)),
      tuple(max(x, y) for x, y in zip(a.stop, b.stop)),
      a.step,
  )


def merge_regions(regions: Sequence[Region], itemsize: int) -> list[Region]:
  """Merges reads of one source that sit close together.

  When requests sit on different lattices and one dense rectangle around all
  of them stays within the slack allowance, that single read serves them all;
  interleaved strided requests that tile a block, such as a rotary reorder,
  land here. Otherwise regions on the same lattice are visited in order of
  their start, and each joins the previous merged read when the combined
  rectangle stays within the allowance. Overlapping and duplicate reads always
  merge.

  Args:
    regions: The read requests for one source.
    itemsize: Bytes per element of that source.

  Returns:
    The merged reads; every request lies inside exactly one of them.
  """

  def fits(box: Region, needed: int) -> bool:
    waste = (box.size - needed) * itemsize
    return waste <= max(
        MERGE_SLACK_FRACTION * needed * itemsize, MERGE_SLACK_BYTES
    )

  def lattice(r: Region) -> tuple[Shape, Shape]:
    return r.steps, tuple(lo % st for lo, st in zip(r.start, r.steps))

  unique = set(regions)
  if len({lattice(r) for r in unique}) > 1:
    dense = Region(
        tuple(min(r.start[d] for r in unique) for d in range(regions[0].ndim)),
        tuple(max(r.stop[d] for r in unique) for d in range(regions[0].ndim)),
    )
    if fits(dense, sum(r.size for r in unique)):
      return [dense]
  merged: list[tuple[Region, int]] = []
  for region in sorted(unique, key=lambda r: (lattice(r), r.start)):
    if merged:
      box, useful = merged[-1]
      union = _bounding(box, region)
      if union is not None and fits(union, useful + region.size):
        merged[-1] = (union, useful + region.size)
        continue
    merged.append((region, region.size))
  return [box for box, _ in merged]


def _serving_slices(outer: Region, inner: Region) -> tuple[slice, ...] | None:
  """Returns how to cut `inner` out of a read of `outer`, if it can be."""
  if outer.ndim != inner.ndim:
    return None
  slices = []
  for lo_o, lo_i, n_o, n_i, s_o, s_i in zip(
      outer.start,
      inner.start,
      outer.shape,
      inner.shape,
      outer.steps,
      inner.steps,
  ):
    offset = lo_i - lo_o
    if offset < 0 or offset % s_o or s_i % s_o:
      return None
    first, step = offset // s_o, s_i // s_o
    if first + (n_i - 1) * step >= n_o:
      return None
    slices.append(slice(first, first + (n_i - 1) * step + 1, step))
  return tuple(slices)


@dataclasses.dataclass
class LoadStats:
  """Where the time of a load goes, summed over every shard it fills.

  Attributes:
    read_seconds: Time inside storage reads, summed over the threads that
      fill shards in parallel.
    read_bytes: Bytes the storage reads returned.
    reads: Number of storage reads issued.
    host_seconds: Time laying values out and copying them into shard buffers,
      per-element math included, reads excluded; summed over threads.
    transfer_seconds: Wall time spent waiting for host-to-device transfers
      after every shard buffer was filled; transfers that overlap host work
      do not count.
    total_seconds: Wall time of the load.
    shards: Number of shard buffers filled.
  """

  read_seconds: float = 0.0
  read_bytes: int = 0
  reads: int = 0
  host_seconds: float = 0.0
  transfer_seconds: float = 0.0
  total_seconds: float = 0.0
  shards: int = 0
  _lock: threading.Lock = dataclasses.field(
      default_factory=threading.Lock, repr=False, compare=False
  )

  def add_read(self, seconds: float, nbytes: int) -> None:
    """Records one storage read; safe to call from several threads."""
    with self._lock:
      self.read_seconds += seconds
      self.read_bytes += nbytes
      self.reads += 1


class ReadBatch:
  """Serves a known set of source reads with as few storage calls as possible.

  Requests are grouped by source and merged with `merge_regions`. Each merged
  read is issued on first use and every request inside it is then a slice of
  the result.
  """

  def __init__(
      self,
      requests: Iterable[tuple[LeafRef, Region]],
      stats: LoadStats | None = None,
  ):
    self._stats = stats
    groups: dict[int, tuple[LeafRef, list[Region]]] = {}
    for ref, region in requests:
      groups.setdefault(id(ref), (ref, []))[1].append(region)
    self._blocks: dict[int, list[list[Any]]] = {
        key: [
            [box, None, threading.Lock()]
            for box in merge_regions(regions, np.dtype(ref.dtype).itemsize)
        ]
        for key, (ref, regions) in groups.items()
    }

  def fetch(self, ref: LeafRef, region: Region) -> np.ndarray:
    """Returns `region` of `ref`, reading its merged block if not yet read."""
    for entry in self._blocks.get(id(ref), ()):
      box, _, lock = entry
      cut = _serving_slices(box, region)
      if cut is None:
        continue
      # Boxes of one shard fill on several threads; the first to need a
      # merged block reads it and the others wait for that read.
      with lock:
        if entry[1] is None:
          entry[1] = self._read(ref, box)
      return entry[1][cut]
    return self._read(ref, region)

  def _read(self, ref: LeafRef, region: Region) -> np.ndarray:
    start = time.perf_counter()
    block = np.asarray(ref.read(region))
    if self._stats is not None:
      self._stats.add_read(time.perf_counter() - start, block.nbytes)
    return block
