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


"""How each JAX primitive becomes boxes.

A `Rules` table drives the interpreter. Each primitive is one of:

* a layout primitive, whose rule rewrites operand boxes into result boxes and
  never touches data;
* an elementwise primitive, kept inside each box and evaluated per shard;
* a call primitive, whose inner jaxpr is interpreted in place;
* anything else, computed once from its full inputs as a derived source.

`DEFAULT_RULES` covers the primitives conversion functions use. A caller can
add a rule for another primitive, replace one, or remove one so that primitive
is computed whole, and pass the result to `load` or `preview`. The public
functions below are the building blocks a rule composes: they take and return
`VirtualLeaf` values and raise `Unsupported` to fall back to whole computation.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
import dataclasses
import itertools
import math
from typing import Any

from jax.extend import core as jax_core
import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import boxes

Region = boxes.Region
Assignment = boxes.Assignment
VirtualLeaf = boxes.VirtualLeaf
Const = boxes.Const
Apply = boxes.Apply
Expr = boxes.Expr
Shape = boxes.Shape
substitute = boxes.substitute

# Maps (operands, params, result shape, result dtype) to one value per result.
LayoutRule = Callable[
    [Sequence[VirtualLeaf], Mapping[str, Any], Shape, np.dtype],
    list[VirtualLeaf],
]


# Past this many boxes in one value, per-box bookkeeping costs more than the
# reads it saves, so the producing primitive is computed whole instead.
_MAX_PIECES = 1 << 16

_CALL_PRIMITIVES = frozenset(
    {
        "pjit",
        "jit",
        "closed_call",
        "core_call",
        "custom_jvp_call",
        "custom_vjp_call",
        "custom_vjp_call_jaxpr",
        "remat",
        "checkpoint",
    }
)
_ELEMENTWISE_PRIMITIVES = frozenset(
    {
        "abs",
        "add",
        "and",
        "cbrt",
        "ceil",
        "clamp",
        "convert_element_type",
        "cos",
        "div",
        "eq",
        "exp",
        "exp2",
        "expm1",
        "floor",
        "ge",
        "gt",
        "integer_pow",
        "is_finite",
        "le",
        "log",
        "log1p",
        "logistic",
        "lt",
        "max",
        "min",
        "mul",
        "ne",
        "neg",
        "nextafter",
        "not",
        "or",
        "pow",
        "reduce_precision",
        "rem",
        "round",
        "rsqrt",
        "select_n",
        "sign",
        "sin",
        "sqrt",
        "square",
        "sub",
        "tanh",
        "xor",
    }
)


class Unsupported(Exception):
  """A rule cannot express these operands as boxes; compute the result whole."""


class TooManyPieces(Exception):
  """A symbolic rule would split a value past `_MAX_PIECES` boxes."""


def _identity(n: int) -> np.ndarray:
  return np.eye(n, dtype=np.int64)


def _zeros(n: int) -> np.ndarray:
  return np.zeros(n, dtype=np.int64)


def _ceil_div(a: int, b: int) -> int:
  return -((-a) // b)


def scalar(value: np.ndarray) -> VirtualLeaf:
  return VirtualLeaf(
      (), value.dtype, (Assignment(Region.full(()), Const(value)),)
  )


def _rebox(
    value: VirtualLeaf,
    shape: Sequence[int],
    dtype: np.dtype,
    rule: Callable[[Assignment], Iterator[Assignment]],
) -> VirtualLeaf:
  boxes = tuple(
      itertools.chain.from_iterable(rule(a) for a in value.assignments)
  )
  if len(boxes) > _MAX_PIECES:
    raise TooManyPieces()
  return VirtualLeaf(tuple(shape), np.dtype(dtype), boxes)


def _narrow(box: Assignment, axis: int, lo: int, hi: int) -> Assignment:
  """Restricts a box to `[lo, hi)` along `axis`, in target coordinates."""
  region = box.target_region
  offset = _zeros(region.ndim)
  offset[axis] = lo - region.start[axis]
  return Assignment(
      Region(
          region.start[:axis] + (lo,) + region.start[axis + 1 :],
          region.stop[:axis] + (hi,) + region.stop[axis + 1 :],
      ),
      substitute(box.expr, offset, _identity(region.ndim)),
  )


def _map_axes(
    value: VirtualLeaf, shape: Sequence[int], sources: Sequence[int | None]
) -> VirtualLeaf:
  """Builds each result axis from an operand axis or as a broadcast.

  Result axis `o` copies operand axis `sources[o]` when their sizes agree.
  Otherwise (no source, or a unit axis broadcast) every box spans the whole
  result extent and every view stands still along it. Operand axes no result
  axis takes must have size one. Transpose, squeeze, expand and broadcast are
  all this one rule.

  Args:
    value: The operand.
    shape: The result shape.
    sources: For each result axis, the operand axis it copies, or `None`.

  Returns:
    The result, with no box split.
  """
  copied = [
      (o, i)
      for o, i in enumerate(sources)
      if i is not None and value.shape[i] == shape[o]
  ]
  linear = np.zeros((len(value.shape), len(shape)), dtype=np.int64)
  for o, i in copied:
    linear[i, o] = 1

  def rule(box):
    start, stop = [0] * len(shape), list(shape)
    for o, i in copied:
      start[o] = box.target_region.start[i]
      stop[o] = box.target_region.stop[i]
    yield Assignment(
        Region(tuple(start), tuple(stop)),
        substitute(box.expr, _zeros(len(value.shape)), linear),
    )

  return _rebox(value, shape, value.dtype, rule)


def transpose(value: VirtualLeaf, perm: Sequence[int]) -> VirtualLeaf:
  return _map_axes(value, [value.shape[p] for p in perm], perm)


def squeeze(value: VirtualLeaf, dimensions: Sequence[int]) -> VirtualLeaf:
  kept = [d for d in range(len(value.shape)) if d not in set(dimensions)]
  return _map_axes(value, [value.shape[d] for d in kept], kept)


def expand_dims(value: VirtualLeaf, unit_axes: Sequence[int]) -> VirtualLeaf:
  operand_axes = iter(range(len(value.shape)))
  sources = [
      None if d in set(unit_axes) else next(operand_axes)
      for d in range(len(value.shape) + len(unit_axes))
  ]
  shape = [1 if s is None else value.shape[s] for s in sources]
  return _map_axes(value, shape, sources)


def broadcast_in_dim(
    value: VirtualLeaf,
    shape: Sequence[int],
    broadcast_dimensions: Sequence[int],
) -> VirtualLeaf:
  sources: list[int | None] = [None] * len(shape)
  for operand_axis, result_axis in enumerate(broadcast_dimensions):
    sources[result_axis] = operand_axis
  return _map_axes(value, shape, sources)


def strided_slice(
    value: VirtualLeaf,
    starts: Sequence[int],
    limits: Sequence[int],
    strides: Sequence[int] | None,
) -> VirtualLeaf:
  """Clips each box to the selected progression; strides scale views."""
  n = len(value.shape)
  steps = tuple(strides) if strides else (1,) * n
  shape = tuple(
      max(0, _ceil_div(hi - lo, st))
      for lo, hi, st in zip(starts, limits, steps)
  )
  linear = np.diag(np.asarray(steps, dtype=np.int64)).reshape(n, n)

  def rule(box):
    region = box.target_region
    lo, hi, offset = [], [], []
    for d in range(n):
      o_lo = max(0, _ceil_div(region.start[d] - starts[d], steps[d]))
      o_hi = min(shape[d], _ceil_div(region.stop[d] - starts[d], steps[d]))
      if o_hi <= o_lo:
        return
      lo.append(o_lo)
      hi.append(o_hi)
      offset.append(starts[d] + o_lo * steps[d] - region.start[d])
    yield Assignment(
        Region(tuple(lo), tuple(hi)),
        substitute(box.expr, np.asarray(offset, dtype=np.int64), linear),
    )

  return _rebox(value, shape, value.dtype, rule)


def rev(value: VirtualLeaf, dimensions: Sequence[int]) -> VirtualLeaf:
  """Mirrors box bounds and walks each view backwards along `dimensions`."""
  n = len(value.shape)

  def rule(box):
    region = box.target_region
    start, stop = list(region.start), list(region.stop)
    offset, linear = _zeros(n), _identity(n)
    for d in dimensions:
      size = value.shape[d]
      start[d], stop[d] = size - region.stop[d], size - region.start[d]
      offset[d] = region.stop[d] - 1 - region.start[d]
      linear[d, d] = -1
    yield Assignment(
        Region(tuple(start), tuple(stop)), substitute(box.expr, offset, linear)
    )

  return _rebox(value, value.shape, value.dtype, rule)


def _mergeable(expr: Expr, axis: int, inner_size: int) -> bool:
  """True if axes `axis` and `axis + 1` index every view contiguously."""
  return all(
      np.array_equal(v.matrix[:, axis], inner_size * v.matrix[:, axis + 1])
      for v in boxes.views(expr)
  )


def _merge_pair(value: VirtualLeaf, axis: int) -> VirtualLeaf:
  """Merges `axis` with the next axis, splitting boxes where needed.

  A box stays whole when its merged extent is a contiguous run and every view
  steps through the outer axis exactly one inner extent at a time. Otherwise it
  is cut into unit slices along the outer axis, each of which merges trivially.

  Args:
    value: The value to reshape.
    axis: The outer of the two adjacent axes to merge.

  Returns:
    The value with one fewer axis.
  """
  outer_size, inner_size = value.shape[axis], value.shape[axis + 1]
  n = len(value.shape)
  linear = np.zeros((n, n - 1), dtype=np.int64)
  for d in range(n):
    if d < axis:
      linear[d, d] = 1
    elif d == axis + 1:
      linear[d, axis] = 1
    elif d > axis + 1:
      linear[d, d - 1] = 1

  def merged(box):
    region = box.target_region
    lo = region.start[axis] * inner_size + region.start[axis + 1]
    hi = (region.stop[axis] - 1) * inner_size + region.stop[axis + 1]
    return Assignment(
        Region(
            region.start[:axis] + (lo,) + region.start[axis + 2 :],
            region.stop[:axis] + (hi,) + region.stop[axis + 2 :],
        ),
        substitute(box.expr, _zeros(n), linear),
    )

  def rule(box):
    region = box.target_region
    outer = region.stop[axis] - region.start[axis]
    full_inner = (
        region.start[axis + 1] == 0 and region.stop[axis + 1] == inner_size
    )
    if outer == 1 or (full_inner and _mergeable(box.expr, axis, inner_size)):
      yield merged(box)
      return
    if outer > _MAX_PIECES:
      raise TooManyPieces()
    for t in range(region.start[axis], region.stop[axis]):
      yield merged(_narrow(box, axis, t, t + 1))

  shape = (
      value.shape[:axis] + (outer_size * inner_size,) + value.shape[axis + 2 :]
  )
  return _rebox(value, shape, value.dtype, rule)


def _split_linear(n: int, axis: int, step: int) -> np.ndarray:
  """Maps local coordinates after splitting `axis` back to those before.

  The old coordinate along `axis` becomes `step * outer + inner`; every other
  axis passes through, shifted past the new axis.

  Args:
    n: The rank before the split.
    axis: The axis being split.
    step: The inner extent, or 0 when the box spans a single outer index.

  Returns:
    The substitution matrix, shaped (n, n + 1).
  """
  linear = np.zeros((n, n + 1), dtype=np.int64)
  for d in range(n):
    if d < axis:
      linear[d, d] = 1
    elif d == axis:
      linear[d, axis] = step
      linear[d, axis + 1] = 1
    else:
      linear[d, d + 1] = 1
  return linear


def _split_box(box: Assignment, axis: int, rest: int) -> Assignment:
  """Splits one box whose extent along `axis` is a box after the split.

  That holds when the extent is aligned to whole inner runs, or lies within a
  single outer index.

  Args:
    box: The box to split.
    axis: The axis being split into (outer, `rest`).
    rest: The inner extent.

  Returns:
    The same values, indexed with the extra axis.
  """
  region = box.target_region
  lo, hi = region.start[axis], region.stop[axis]
  if lo % rest == 0 and hi % rest == 0:
    outer, inner, step = (lo // rest, hi // rest), (0, rest), rest
  else:
    t = lo // rest
    outer, inner, step = (t, t + 1), (lo - t * rest, hi - t * rest), 0
  return Assignment(
      Region(
          region.start[:axis] + (outer[0], inner[0]) + region.start[axis + 1 :],
          region.stop[:axis] + (outer[1], inner[1]) + region.stop[axis + 1 :],
      ),
      substitute(
          box.expr, _zeros(region.ndim), _split_linear(region.ndim, axis, step)
      ),
  )


def _split_axis(
    value: VirtualLeaf, axis: int, sizes: Sequence[int]
) -> VirtualLeaf:
  """Splits `axis` into `sizes`, outermost first.

  A box that straddles an inner-run boundary without being aligned to it is
  cut at those boundaries first: an unaligned head, an aligned middle, and an
  unaligned tail.

  Args:
    value: The value to reshape.
    axis: The axis to split.
    sizes: The new axis sizes; their product is the old size.

  Returns:
    The value with `axis` replaced by `len(sizes)` axes.
  """
  if len(sizes) == 1:
    return value
  outer, rest = sizes[0], math.prod(sizes[1:])

  def rule(box):
    lo, hi = box.target_region.start[axis], box.target_region.stop[axis]
    if lo // rest == (hi - 1) // rest or (lo % rest == 0 and hi % rest == 0):
      yield _split_box(box, axis, rest)
      return
    head, tail = _ceil_div(lo, rest) * rest, (hi // rest) * rest
    for seg_lo, seg_hi in ((lo, head), (head, tail), (tail, hi)):
      if seg_lo < seg_hi:
        yield _split_box(_narrow(box, axis, seg_lo, seg_hi), axis, rest)

  shape = value.shape[:axis] + (outer, rest) + value.shape[axis + 1 :]
  return _split_axis(
      _rebox(value, shape, value.dtype, rule), axis + 1, sizes[1:]
  )


def _reshape_groups(
    old: Sequence[int], new: Sequence[int]
) -> list[tuple[int, list[int]]]:
  """Pairs runs of non-unit axes with equal products: (old count, new sizes)."""
  groups = []
  i = j = 0
  while i < len(old):
    p_old, p_new = old[i], new[j]
    count, sizes = 1, [new[j]]
    i, j = i + 1, j + 1
    while p_old != p_new:
      if p_old < p_new:
        p_old *= old[i]
        count += 1
        i += 1
      else:
        p_new *= new[j]
        sizes.append(new[j])
        j += 1
    groups.append((count, sizes))
  return groups


def reshape(
    value: VirtualLeaf,
    new_sizes: Sequence[int],
    dimensions: Sequence[int] | None,
) -> VirtualLeaf:
  """Reshapes by merging, then splitting, runs of axes with equal size."""
  if dimensions is not None:
    value = transpose(value, dimensions)
  new_sizes = tuple(new_sizes)
  if math.prod(value.shape) == 0:
    return VirtualLeaf(new_sizes, value.dtype, ())
  value = squeeze(value, [d for d, s in enumerate(value.shape) if s == 1])
  groups = _reshape_groups(value.shape, [s for s in new_sizes if s != 1])
  axis = 0
  for count, _ in groups:
    for inner in range(axis + count - 2, axis - 1, -1):
      value = _merge_pair(value, inner)
    axis += 1
  axis = 0
  for _, sizes in groups:
    value = _split_axis(value, axis, sizes)
    axis += len(sizes)
  return expand_dims(value, [d for d, s in enumerate(new_sizes) if s == 1])


def _shifted(boxes: Sequence[Assignment], offsets: Sequence[int]):
  return [Assignment(b.target_region.shifted(offsets), b.expr) for b in boxes]


def concatenate(
    values: Sequence[VirtualLeaf], dimension: int, shape: Shape, dtype: np.dtype
) -> VirtualLeaf:
  """Offsets each operand's boxes along `dimension`; views are unchanged."""
  boxes = []
  offsets = [0] * len(shape)
  for value in values:
    boxes += _shifted(value.assignments, offsets)
    offsets[dimension] += value.shape[dimension]
  return VirtualLeaf(tuple(shape), np.dtype(dtype), tuple(boxes))


def _fill(scalar: VirtualLeaf, region: Region) -> Assignment:
  """Repeats a scalar value over `region`."""
  (box,) = scalar.assignments
  return Assignment(
      region,
      substitute(box.expr, _zeros(0), np.zeros((0, region.ndim), np.int64)),
  )


def pad(
    value: VirtualLeaf,
    padding: VirtualLeaf,
    config: Sequence[tuple[int, int, int]],
    shape: Shape,
) -> VirtualLeaf:
  """Shifts boxes inward and fills the border with the padding scalar."""
  if any(interior != 0 for _, _, interior in config):
    raise Unsupported()
  n = len(value.shape)
  crop_lo = [max(0, -int(lo)) for lo, _, _ in config]
  crop_hi = [s - max(0, -int(hi)) for s, (_, hi, _) in zip(value.shape, config)]
  if any(crop_lo) or any(c != s for c, s in zip(crop_hi, value.shape)):
    value = strided_slice(value, crop_lo, crop_hi, None)
  inner_lo = tuple(max(0, int(lo)) for lo, _, _ in config)
  inner_hi = tuple(s + w for s, w in zip(inner_lo, value.shape))
  boxes = _shifted(value.assignments, inner_lo)
  for d in range(n):
    for lo, hi in ((0, inner_lo[d]), (inner_hi[d], shape[d])):
      if lo < hi:
        start = inner_lo[:d] + (lo,) + (0,) * (n - d - 1)
        stop = inner_hi[:d] + (hi,) + tuple(shape[d + 1 :])
        boxes.append(_fill(padding, Region(start, stop)))
  return VirtualLeaf(tuple(shape), value.dtype, tuple(boxes))


def _scalar_int(value: VirtualLeaf) -> int:
  """Returns the integer a scalar value holds, if it is a known constant."""
  if value.shape != () or len(value.assignments) != 1:
    raise Unsupported()
  expr = value.assignments[0].expr
  if not isinstance(expr, Const):
    raise Unsupported()
  return int(expr.value)


def _dynamic_slice(
    value: VirtualLeaf,
    indices: Sequence[VirtualLeaf],
    slice_sizes: Sequence[int],
) -> VirtualLeaf:
  starts = [
      min(max(_scalar_int(index), 0), dim - size)
      for index, size, dim in zip(indices, slice_sizes, value.shape)
  ]
  limits = [s + size for s, size in zip(starts, slice_sizes)]
  return strided_slice(value, starts, limits, None)


def _restrict(expr: Expr, outer: Region, inner: Region) -> Expr:
  """Re-expresses an expression over `outer` as one over its sub-box `inner`."""
  offset = np.asarray(inner.start, dtype=np.int64) - np.asarray(
      outer.start, dtype=np.int64
  )
  return substitute(expr, offset, _identity(inner.ndim))


def elementwise(
    primitive: jax_core.Primitive,
    params: dict[str, Any],
    values: Sequence[VirtualLeaf],
    shape: Shape,
    dtype: np.dtype,
) -> VirtualLeaf:
  """Aligns operand boxes and wraps each common box in an `Apply`."""
  operands = []
  for value in values:
    if value.shape == () and shape != ():
      value = VirtualLeaf(
          shape, value.dtype, (_fill(value, Region.full(shape)),)
      )
    elif value.shape != shape:
      # Size-1 axes broadcast implicitly against the result.
      if len(value.shape) != len(shape):
        raise Unsupported()
      value = broadcast_in_dim(value, shape, range(len(shape)))
    operands.append(value)
  cells = [(a.target_region, [a.expr]) for a in operands[0].assignments]
  for value in operands[1:]:
    refined = []
    for region, exprs in cells:
      for box in value.assignments:
        hit = region.intersect(box.target_region)
        if hit is None:
          continue
        refined.append(
            (
                hit,
                [_restrict(e, region, hit) for e in exprs]
                + [_restrict(box.expr, box.target_region, hit)],
            )
        )
        if len(refined) > _MAX_PIECES:
          raise TooManyPieces()
    cells = refined
  return VirtualLeaf(
      tuple(shape),
      np.dtype(dtype),
      tuple(
          Assignment(region, Apply(primitive, params, tuple(exprs)))
          for region, exprs in cells
      ),
  )


def slices_along(
    x: VirtualLeaf, axis: int, bounds: Sequence[tuple[int, int]]
) -> list[VirtualLeaf]:
  out = []
  for lo, hi in bounds:
    starts, limits = [0] * len(x.shape), list(x.shape)
    starts[axis], limits[axis] = lo, hi
    out.append(strided_slice(x, starts, limits, None))
  return out


def _split_rule(ops, params, shape, dtype) -> list[VirtualLeaf]:
  del shape, dtype
  edges = np.cumsum([0, *params["sizes"]])
  return slices_along(ops[0], params["axis"], list(zip(edges, edges[1:])))


def _unstack_rule(ops, params, shape, dtype) -> list[VirtualLeaf]:
  del shape, dtype
  axis = params["axis"]
  bounds = [(i, i + 1) for i in range(ops[0].shape[axis])]
  return [squeeze(v, [axis]) for v in slices_along(ops[0], axis, bounds)]


def _stack_rule(ops, params, shape, dtype) -> list[VirtualLeaf]:
  axis = params["axis"]
  return [
      concatenate([expand_dims(v, [axis]) for v in ops], axis, shape, dtype)
  ]


def _tile_rule(ops, params, shape, dtype) -> list[VirtualLeaf]:
  del shape, dtype
  tiled = ops[0]
  for axis, reps in enumerate(params["reps"]):
    grown = list(tiled.shape)
    grown[axis] *= reps
    tiled = concatenate([tiled] * reps, axis, tuple(grown), tiled.dtype)
  return [tiled]


_DEFAULT_LAYOUT: dict[str, LayoutRule] = {
    "transpose": lambda ops, p, s, d: [transpose(ops[0], p["permutation"])],
    "reshape": lambda ops, p, s, d: [
        reshape(ops[0], p["new_sizes"], p.get("dimensions"))
    ],
    "squeeze": lambda ops, p, s, d: [squeeze(ops[0], p["dimensions"])],
    "broadcast_in_dim": lambda ops, p, s, d: [
        broadcast_in_dim(ops[0], p["shape"], p["broadcast_dimensions"])
    ],
    "slice": lambda ops, p, s, d: [
        strided_slice(
            ops[0], p["start_indices"], p["limit_indices"], p.get("strides")
        )
    ],
    "rev": lambda ops, p, s, d: [rev(ops[0], p["dimensions"])],
    "concatenate": lambda ops, p, s, d: [
        concatenate(ops, p["dimension"], s, d)
    ],
    "pad": lambda ops, p, s, d: [pad(ops[0], ops[1], p["padding_config"], s)],
    "dynamic_slice": lambda ops, p, s, d: [
        _dynamic_slice(ops[0], ops[1:], p["slice_sizes"])
    ],
    "copy": lambda ops, p, s, d: [ops[0]],
    "copy_p": lambda ops, p, s, d: [ops[0]],
    "split": _split_rule,
    "unstack": _unstack_rule,
    "stack": _stack_rule,
    "tile": _tile_rule,
}


@dataclasses.dataclass(frozen=True)
class Rules:
  """How the interpreter treats each primitive, by name.

  Attributes:
    layout: Rules for primitives that only move data.
    elementwise: Primitives evaluated per element, inside each box.
    calls: Primitives that carry an inner jaxpr to interpret in place.
  """

  layout: Mapping[str, LayoutRule]
  elementwise: frozenset[str]
  calls: frozenset[str]

  def with_layout(self, name: str, rule: LayoutRule) -> "Rules":
    """Returns these rules with `rule` handling primitive `name`."""
    return dataclasses.replace(self, layout={**self.layout, name: rule})

  def with_elementwise(self, *names: str) -> "Rules":
    """Returns these rules with `names` evaluated per element."""
    return dataclasses.replace(self, elementwise=self.elementwise | set(names))

  def without(self, *names: str) -> "Rules":
    """Returns these rules with `names` computed whole, as derived sources."""
    return Rules(
        layout={k: v for k, v in self.layout.items() if k not in names},
        elementwise=self.elementwise - set(names),
        calls=self.calls - set(names),
    )


DEFAULT_RULES = Rules(
    layout=_DEFAULT_LAYOUT,
    elementwise=_ELEMENTWISE_PRIMITIVES,
    calls=_CALL_PRIMITIVES,
)
