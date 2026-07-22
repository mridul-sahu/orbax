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

"""Operations that transform a manifest.

Each operation is a factory returning a pure function of a manifest and a
resolution context. Operations move regions, never data, so applying one is a
symbolic rewrite that reads no array bytes. Defects do not raise; they
accumulate in the context's report, so one preview returns the complete repair
list rather than the first item of it.
"""

import collections
from collections.abc import Callable, Sequence
import dataclasses
import re

import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import manifest as manifest_lib
from orbax.checkpoint.experimental.v1._src.surgery import report as report_lib

Manifest = manifest_lib.Manifest
Leaf = manifest_lib.Leaf
LeafRef = manifest_lib.LeafRef
VirtualLeaf = manifest_lib.VirtualLeaf
Assignment = manifest_lib.Assignment
Region = manifest_lib.Region


@dataclasses.dataclass
class ResolveContext:
  """Ambient state threaded through a resolution.

  Attributes:
    on_missing: How to treat a target leaf the pipeline does not produce,
      either "error" or "init_from_target".
    report: Accumulates the account and every defect found.
    source_refs: Per-namespace source leaves, keyed by namespace then key. Used
      by `take` to import from a named source.
  """

  on_missing: str
  report: report_lib.PlanReport
  source_refs: dict[str, dict[str, LeafRef]] = dataclasses.field(
      default_factory=dict
  )


Op = Callable[[Manifest, ResolveContext], Manifest]


def _const_init(value: float) -> manifest_lib.InitFn:
  def init(shape: manifest_lib.Shape, dtype: np.dtype) -> np.ndarray:
    return np.full(shape, value, dtype=dtype)

  return init


def _shift_target(a: Assignment, axis: int, offset: int) -> Assignment:
  """Returns `a` with its target region shifted by `offset` along `axis`."""
  tr = a.target_region
  start = tr.start[:axis] + (tr.start[axis] + offset,) + tr.start[axis + 1 :]
  stop = tr.stop[:axis] + (tr.stop[axis] + offset,) + tr.stop[axis + 1 :]
  return dataclasses.replace(a, target_region=Region(start, stop))


def rename(*rules: tuple[str, str]) -> Op:
  r"""Rewrites keys by applying ordered regex replacement rules.

  Every rule is applied in order to every key. A rule that matches no key is a
  likely typo and is reported as an error.

  Args:
    *rules: (pattern, replacement) pairs, applied left to right.

  Returns:
    An operation that renames matching keys.
  """
  compiled = [(re.compile(p), r) for p, r in rules]

  def op(manifest: Manifest, ctx: ResolveContext) -> Manifest:
    fired = [False] * len(compiled)
    result: Manifest = {}
    for key, leaf in manifest.items():
      new_key = key
      for i, (pattern, replacement) in enumerate(compiled):
        if pattern.search(new_key):
          fired[i] = True
          new_key = pattern.sub(replacement, new_key)
      if new_key != key:
        ctx.report.moved.append((key, new_key))
      if new_key in result:
        ctx.report.add_error(
            report_lib.RENAME_COLLISION,
            new_key,
            f"multiple keys rename to {new_key!r}",
        )
      result[new_key] = leaf
    for i, did_fire in enumerate(fired):
      if not did_fire:
        ctx.report.add_error(
            report_lib.UNMATCHED_RULE,
            rules[i][0],
            f"rename rule {rules[i][0]!r} matched no key",
        )
    return result

  return op


def drop(pattern: str) -> Op:
  """Removes keys matching `pattern`; they are never read.

  Args:
    pattern: Regex matched against each key.

  Returns:
    An operation that drops matching keys.
  """
  compiled = re.compile(pattern)

  def op(manifest: Manifest, ctx: ResolveContext) -> Manifest:
    result: Manifest = {}
    for key, leaf in manifest.items():
      if compiled.search(key):
        ctx.report.dropped.append(key)
      else:
        result[key] = leaf
    return result

  return op


def select(pattern: str) -> Op:
  """Keeps only keys matching `pattern`; the rest are dropped.

  Args:
    pattern: Regex matched against each key.

  Returns:
    An operation that keeps matching keys and drops the rest.
  """
  compiled = re.compile(pattern)

  def op(manifest: Manifest, ctx: ResolveContext) -> Manifest:
    result: Manifest = {}
    matched_any = False
    for key, leaf in manifest.items():
      if compiled.search(key):
        result[key] = leaf
        matched_any = True
      else:
        ctx.report.dropped.append(key)
    if not matched_any:
      ctx.report.add_error(
          report_lib.UNMATCHED_RULE,
          pattern,
          f"select pattern {pattern!r} matched no key",
      )
    return result

  return op


def _build_stack(
    base: str,
    idx_leaves: dict[int, tuple[str, Leaf]],
    axis: int,
    filler: float | None,
    ctx: ResolveContext,
) -> VirtualLeaf:
  """Assembles one stacked virtual leaf from its indexed parts."""
  indices = sorted(idx_leaves)
  count = indices[-1] + 1
  rep = manifest_lib.as_virtual(idx_leaves[indices[0]][1])
  base_shape = rep.shape
  target_shape = base_shape[:axis] + (count,) + base_shape[axis:]

  assignments: list[Assignment] = []
  sources: dict[int, str] = {}
  for i in indices:
    key_i, leaf_i = idx_leaves[i]
    sources[i] = key_i
    v = manifest_lib.as_virtual(leaf_i)
    if v.shape != base_shape:
      ctx.report.add_error(
          report_lib.SHAPE_MISMATCH,
          key_i,
          f"stack {base!r} index {i} has shape {v.shape}, expected {base_shape}",
      )
      continue
    for a in v.assignments:
      if a.inserted_axis is not None or a.sliced_source_axis is not None:
        ctx.report.add_error(
            report_lib.INCOMPLETE_GROUP,
            key_i,
            f"stack {base!r} cannot compose over a rank-changing source",
        )
        continue
      new_target = a.target_region.with_inserted_axis(axis, i)
      assignments.append(
          dataclasses.replace(
              a, target_region=new_target, inserted_axis=axis
          )
      )

  ctx.report.stacked[base] = sources
  missing = [i for i in range(count) if i not in idx_leaves]
  init = None
  if missing:
    if filler is None:
      ctx.report.add_error(
          report_lib.INCOMPLETE_GROUP,
          base,
          f"stack {base!r} missing indices {missing}; provide a filler",
      )
    else:
      init = _const_init(filler)
      ctx.report.filled[base] = f"stack padding for indices {missing}"
  return VirtualLeaf(
      shape=target_shape,
      dtype=rep.dtype,
      assignments=tuple(assignments),
      init=init,
  )


def stack(
    pattern: str, *, axis: int = 0, filler: float | None = None
) -> Op:
  r"""Collapses keys matching a single-index pattern into one stacked leaf.

  The pattern must contain exactly one capture group holding a positive
  integer. That group is the stack index and is removed to form the base key.
  A gap in the indices without a filler is an error.

  Args:
    pattern: Regex with one integer capture group.
    axis: Axis the new stacked dimension is inserted at.
    filler: Value for missing indices, or `None` to require a complete range.

  Returns:
    An operation that stacks matching keys.
  """
  compiled = re.compile(pattern)

  def op(manifest: Manifest, ctx: ResolveContext) -> Manifest:
    groups: dict[str, dict[int, tuple[str, Leaf]]] = collections.defaultdict(
        dict
    )
    order: list[str] = []
    result: Manifest = {}
    matched_any = False
    for key, leaf in manifest.items():
      m = compiled.search(key)
      if not m:
        result[key] = leaf
        continue
      if len(m.groups()) != 1:
        ctx.report.add_error(
            report_lib.INCOMPLETE_GROUP,
            key,
            f"stack pattern {pattern!r} must have exactly one capture group",
        )
        result[key] = leaf
        continue
      digits = re.findall(r"\d+", m.group(1))
      if len(digits) != 1:
        ctx.report.add_error(
            report_lib.INCOMPLETE_GROUP,
            key,
            f"stack index must be a single integer, got {m.group(1)!r}",
        )
        result[key] = leaf
        continue
      matched_any = True
      idx = int(digits[0])
      base = key[: m.start(1)] + key[m.end(1) :]
      if base not in groups:
        order.append(base)
      groups[base][idx] = (key, leaf)
    if not matched_any:
      ctx.report.add_error(
          report_lib.UNMATCHED_RULE,
          pattern,
          f"stack pattern {pattern!r} matched no key",
      )
    for base in order:
      result[base] = _build_stack(base, groups[base], axis, filler, ctx)
    return result

  return op


def unstack(pattern: str, *, axis: int = 0, sep: str = ".") -> Op:
  """Splits each matching leaf into one sliced leaf per index on `axis`.

  New keys are the matched key followed by the separator and the index. This
  does not restore the exact key layout a prior `stack` removed; recovering
  original names is left to a rename.

  Args:
    pattern: Regex matched against each key.
    axis: Axis to slice into individual leaves.
    sep: Separator between the original key and the appended index.

  Returns:
    An operation that unstacks matching leaves.
  """
  compiled = re.compile(pattern)

  def op(manifest: Manifest, ctx: ResolveContext) -> Manifest:
    result: Manifest = {}
    matched_any = False
    for key, leaf in manifest.items():
      if not compiled.search(key):
        result[key] = leaf
        continue
      matched_any = True
      v = manifest_lib.as_virtual(leaf)
      n = v.shape[axis]
      sub_shape = v.shape[:axis] + v.shape[axis + 1 :]
      for i in range(n):
        assignments = []
        for a in v.assignments:
          sliced = _unstack_slice(a, axis, i, key, ctx)
          if sliced is not None:
            assignments.append(sliced)
        result[f"{key}{sep}{i}"] = VirtualLeaf(
            shape=sub_shape,
            dtype=v.dtype,
            assignments=tuple(assignments),
            init=None,
        )
    if not matched_any:
      ctx.report.add_error(
          report_lib.UNMATCHED_RULE,
          pattern,
          f"unstack pattern {pattern!r} matched no key",
      )
    return result

  return op


def _unstack_slice(
    a: Assignment, axis: int, i: int, key: str, ctx: ResolveContext
) -> Assignment | None:
  """Returns the assignment reading source index `i` off `axis`, squeezed."""
  if a.inserted_axis is not None or a.sliced_source_axis is not None:
    ctx.report.add_error(
        report_lib.INCOMPLETE_GROUP,
        key,
        f"unstack cannot compose over a rank-changing source at {key!r}",
    )
    return None
  tr = a.target_region
  if not (tr.start[axis] <= i < tr.stop[axis]):
    return None
  sr = a.source_region
  src_i = sr.start[axis] + (i - tr.start[axis])
  new_sr = Region(
      sr.start[:axis] + (src_i,) + sr.start[axis + 1 :],
      sr.stop[:axis] + (src_i + 1,) + sr.stop[axis + 1 :],
  )
  return dataclasses.replace(
      a,
      target_region=tr.without_axis(axis),
      source_region=new_sr,
      sliced_source_axis=axis,
  )


def _build_fuse(
    part_map: dict[str, tuple[str, Leaf]],
    parts: Sequence[str],
    axis: int,
) -> tuple[VirtualLeaf, list[str]]:
  """Concatenates the ordered parts of one group into a single leaf."""
  offset = 0
  assignments: list[Assignment] = []
  ordered_keys: list[str] = []
  ref_shape: list[int] | None = None
  dtype = None
  for p in parts:
    key, leaf = part_map[p]
    v = manifest_lib.as_virtual(leaf)
    if ref_shape is None:
      ref_shape = list(v.shape)
      dtype = v.dtype
    for a in v.assignments:
      assignments.append(_shift_target(a, axis, offset))
    ordered_keys.append(key)
    offset += v.shape[axis]
  target_shape = (
      tuple(ref_shape[:axis]) + (offset,) + tuple(ref_shape[axis + 1 :])
  )
  return (
      VirtualLeaf(
          shape=target_shape,
          dtype=dtype,
          assignments=tuple(assignments),
          init=None,
      ),
      ordered_keys,
  )


def fuse(*, parts: Sequence[str], into: str, axis: int = 0) -> Op:
  r"""Concatenates keys that differ only by one of `parts` into a fused key.

  Keys sharing a base (the key with the matched part removed) form a group.
  A group missing any part is an error.

  Args:
    parts: Ordered substrings that distinguish the keys to concatenate.
    into: Substring that replaces the part in the fused key.
    axis: Axis to concatenate along.

  Returns:
    An operation that fuses matching groups.
  """
  parts = tuple(parts)

  def op(manifest: Manifest, ctx: ResolveContext) -> Manifest:
    groups: dict[str, dict[str, tuple[str, Leaf]]] = {}
    order: list[str] = []
    consumed: set[str] = set()
    for key, leaf in manifest.items():
      matched_part = next((p for p in parts if p in key), None)
      if matched_part is None:
        continue
      base = key.replace(matched_part, "\x00", 1)
      if base not in groups:
        groups[base] = {}
        order.append(base)
      groups[base][matched_part] = (key, leaf)
      consumed.add(key)

    result: Manifest = {
        key: leaf for key, leaf in manifest.items() if key not in consumed
    }
    for base in order:
      part_map = groups[base]
      if len(part_map) != len(parts):
        missing = [p for p in parts if p not in part_map]
        ctx.report.add_error(
            report_lib.INCOMPLETE_GROUP,
            base.replace("\x00", into),
            f"fuse group missing parts {missing}",
        )
        for key, leaf in part_map.values():
          result[key] = leaf
        continue
      fused_key = base.replace("\x00", into)
      built, ordered_keys = _build_fuse(part_map, parts, axis)
      result[fused_key] = built
      ctx.report.fused[fused_key] = ordered_keys
    if not groups:
      ctx.report.add_error(
          report_lib.UNMATCHED_RULE,
          str(parts),
          f"fuse parts {parts} matched no key",
      )
    return result

  return op


def split(
    *, key: str, sizes: Sequence[int], into: Sequence[str], axis: int = 0
) -> Op:
  """Slices one leaf into several along `axis` at the given sizes.

  Args:
    key: The leaf to split.
    sizes: Extent of each output along `axis`; must sum to the axis size.
    into: New key for each output, in order.
    axis: Axis to slice along.

  Returns:
    An operation that splits the leaf.
  """
  sizes = tuple(sizes)
  into = tuple(into)

  def op(manifest: Manifest, ctx: ResolveContext) -> Manifest:
    result: Manifest = dict(manifest)
    if key not in manifest:
      ctx.report.add_error(
          report_lib.MISSING_SOURCE, key, f"split source {key!r} not present"
      )
      return result
    if len(sizes) != len(into):
      ctx.report.add_error(
          report_lib.SHAPE_MISMATCH,
          key,
          f"split has {len(sizes)} sizes but {len(into)} target keys",
      )
      return result
    leaf = result.pop(key)
    v = manifest_lib.as_virtual(leaf)
    total = v.shape[axis]
    if sum(sizes) != total:
      ctx.report.add_error(
          report_lib.SHAPE_MISMATCH,
          key,
          f"split sizes sum to {sum(sizes)} but axis {axis} is {total}",
      )
      result[key] = leaf
      return result
    offset = 0
    for size, new_key in zip(sizes, into):
      sub_shape = v.shape[:axis] + (size,) + v.shape[axis + 1 :]
      assignments = []
      for a in v.assignments:
        sliced = _slice_axis(a, axis, offset, offset + size)
        if sliced is not None:
          assignments.append(sliced)
      result[new_key] = VirtualLeaf(
          shape=sub_shape,
          dtype=v.dtype,
          assignments=tuple(assignments),
          init=None,
      )
      offset += size
    return result

  return op


def _slice_axis(
    a: Assignment, axis: int, lo: int, hi: int
) -> Assignment | None:
  """Returns the [lo, hi) slice of `a` on `axis`, rebased to a zero origin."""
  tr = a.target_region
  new_lo = max(tr.start[axis], lo)
  new_hi = min(tr.stop[axis], hi)
  if new_hi <= new_lo:
    return None
  t_start = tr.start[:axis] + (new_lo - lo,) + tr.start[axis + 1 :]
  t_stop = tr.stop[:axis] + (new_hi - lo,) + tr.stop[axis + 1 :]
  sr = a.source_region
  shift = new_lo - tr.start[axis]
  extent = new_hi - new_lo
  s_start = sr.start[:axis] + (sr.start[axis] + shift,) + sr.start[axis + 1 :]
  s_stop = (
      sr.start[:axis] + (sr.start[axis] + shift + extent,) + sr.stop[axis + 1 :]
  )
  return dataclasses.replace(
      a,
      target_region=Region(t_start, t_stop),
      source_region=Region(s_start, s_stop),
  )


def repeat(pattern: str, *, axis: int = 0, times: int) -> Op:
  """Tiles each matching leaf `times` times along `axis`.

  The source is read once and placed at each of its destination blocks, so the
  repeat count does not multiply the bytes read.

  Args:
    pattern: Regex matched against each key.
    axis: Axis to tile along.
    times: Number of contiguous copies.

  Returns:
    An operation that repeats matching leaves.
  """
  compiled = re.compile(pattern)

  def op(manifest: Manifest, ctx: ResolveContext) -> Manifest:
    result: Manifest = {}
    matched_any = False
    for key, leaf in manifest.items():
      if not compiled.search(key):
        result[key] = leaf
        continue
      matched_any = True
      v = manifest_lib.as_virtual(leaf)
      block = v.shape[axis]
      assignments = []
      for t in range(times):
        for a in v.assignments:
          assignments.append(_shift_target(a, axis, t * block))
      target_shape = v.shape[:axis] + (block * times,) + v.shape[axis + 1 :]
      result[key] = VirtualLeaf(
          shape=target_shape,
          dtype=v.dtype,
          assignments=tuple(assignments),
          init=None,
      )
    if not matched_any:
      ctx.report.add_error(
          report_lib.UNMATCHED_RULE,
          pattern,
          f"repeat pattern {pattern!r} matched no key",
      )
    return result

  return op


def cast(pattern: str, dtype: np.typing.DTypeLike) -> Op:
  """Converts matching leaves to `dtype` during streaming.

  Args:
    pattern: Regex matched against each key.
    dtype: The destination dtype.

  Returns:
    An operation that casts matching leaves.
  """
  target_dtype = np.dtype(dtype)
  compiled = re.compile(pattern)

  def op(manifest: Manifest, ctx: ResolveContext) -> Manifest:
    result: Manifest = {}
    matched_any = False
    for key, leaf in manifest.items():
      if not compiled.search(key):
        result[key] = leaf
        continue
      matched_any = True
      v = manifest_lib.as_virtual(leaf)
      new_assignments = tuple(
          dataclasses.replace(a, cast=target_dtype) for a in v.assignments
      )
      result[key] = VirtualLeaf(
          shape=v.shape,
          dtype=target_dtype,
          assignments=new_assignments,
          init=v.init,
      )
    if not matched_any:
      ctx.report.add_error(
          report_lib.UNMATCHED_RULE,
          pattern,
          f"cast pattern {pattern!r} matched no key",
      )
    return result

  return op


def resize(
    pattern: str,
    *,
    axis: int = 0,
    size: int,
    init: manifest_lib.InitFn | None = None,
) -> Op:
  """Truncates or grows matching leaves along `axis` to `size`.

  Truncation reads a sub-range of the source. Growth reads the whole source
  into the leading positions and runs `init` over the new tail.

  Args:
    pattern: Regex matched against each key.
    axis: Axis to resize.
    size: New extent along `axis`.
    init: Fills the grown tail; required when `size` exceeds the current extent.

  Returns:
    An operation that resizes matching leaves.
  """
  compiled = re.compile(pattern)

  def op(manifest: Manifest, ctx: ResolveContext) -> Manifest:
    result: Manifest = {}
    matched_any = False
    for key, leaf in manifest.items():
      if not compiled.search(key):
        result[key] = leaf
        continue
      matched_any = True
      v = manifest_lib.as_virtual(leaf)
      cur = v.shape[axis]
      target_shape = v.shape[:axis] + (size,) + v.shape[axis + 1 :]
      if size <= cur:
        assignments = []
        for a in v.assignments:
          sliced = _slice_axis(a, axis, 0, size)
          if sliced is not None:
            assignments.append(sliced)
        result[key] = VirtualLeaf(
            shape=target_shape,
            dtype=v.dtype,
            assignments=tuple(assignments),
            init=None,
        )
      else:
        if init is None:
          ctx.report.add_error(
              report_lib.UNCOVERED_TARGET,
              key,
              f"resize {key!r} grows axis {axis} from {cur} to {size} but no"
              " init was given for the tail",
          )
        else:
          ctx.report.filled[key] = f"resize tail [{cur}:{size}] on axis {axis}"
        result[key] = VirtualLeaf(
            shape=target_shape,
            dtype=v.dtype,
            assignments=v.assignments,
            init=init,
        )
    if not matched_any:
      ctx.report.add_error(
          report_lib.UNMATCHED_RULE,
          pattern,
          f"resize pattern {pattern!r} matched no key",
      )
    return result

  return op


def take(source: str, pattern: str, *, into: str) -> Op:
  r"""Imports keys from a named source, remapping the matched prefix to `into`.

  Args:
    source: The source namespace to draw from.
    pattern: Regex matched against that source's keys.
    into: Replacement for the matched portion of each imported key.

  Returns:
    An operation that imports matching keys into the manifest.
  """
  compiled = re.compile(pattern)

  def op(manifest: Manifest, ctx: ResolveContext) -> Manifest:
    refs = ctx.source_refs.get(source)
    if refs is None:
      ctx.report.add_error(
          report_lib.MISSING_SOURCE,
          source,
          f"take source namespace {source!r} was not provided",
      )
      return manifest
    result: Manifest = dict(manifest)
    matched_any = False
    for key, ref in refs.items():
      if not compiled.search(key):
        continue
      matched_any = True
      new_key = compiled.sub(into, key)
      result[new_key] = ref
      ctx.report.moved.append((f"{source}:{key}", new_key))
    if not matched_any:
      ctx.report.add_error(
          report_lib.UNMATCHED_RULE,
          pattern,
          f"take pattern {pattern!r} matched no key in {source!r}",
      )
    return result

  return op
