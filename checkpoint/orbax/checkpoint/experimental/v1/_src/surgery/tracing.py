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


"""Model surgery written as an ordinary JAX function.

A surgery function such as::

  def surgery(p):
    experts = [p[f"experts.{i}.w"].T for i in range(8)]
    return {"moe.w": jnp.stack(experts).astype(jnp.bfloat16)}

is traced on shape-only inputs built from the source metadata, so nothing is
read, and its jaxpr is interpreted symbolically. Every value becomes a set of
disjoint boxes, and each box carries an expression whose leaves are integer
affine views of source arrays. Layout primitives (transpose, reshape, slice,
concatenate, pad, broadcast and friends) only rewrite those views. Elementwise
primitives stay in the expression and run per shard. Any other primitive becomes
a derived source, computed once from its full inputs and reported as
materialized.

Execution reads, for each device shard, only the source bytes that land in it,
then evaluates elementwise primitives with the same `bind` calls an op-by-op run
of the function makes. The result therefore equals the function applied to the
loaded source under `jax.disable_jit()`, bit for bit.
"""

from collections.abc import Callable, Sequence
import dataclasses
import functools
import itertools
from typing import Any

import jax
from jax.extend import core as jax_core
import numpy as np
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.v1._src.surgery import boxes
from orbax.checkpoint.experimental.v1._src.surgery import fill
from orbax.checkpoint.experimental.v1._src.surgery import host
from orbax.checkpoint.experimental.v1._src.surgery import report as report_lib
from orbax.checkpoint.experimental.v1._src.surgery import rules as rules_lib

LeafRef = boxes.LeafRef
DerivedRef = boxes.DerivedRef
VirtualLeaf = boxes.VirtualLeaf

# The namespace source leaves are filed under; constants and derived values
# use their own, so execution never opens a store for them.
SOURCE = "source"
_CONST_SOURCE = "__const__"

# Parameters under which call primitives carry their inner jaxpr.
_SUBJAXPR_PARAMS = ("jaxpr", "call_jaxpr", "fun_jaxpr")


class _Interpreter:
  """Evaluates a jaxpr over symbolic values, following `rules`."""

  def __init__(self, rules: rules_lib.Rules):
    self._rules = rules
    self._counter = itertools.count()

  def _const(self, value: Any) -> VirtualLeaf:
    array = np.asarray(value)
    if array.ndim == 0:
      return rules_lib.scalar(array)
    return boxes.as_virtual(
        LeafRef(
            source=_CONST_SOURCE,
            key=f"const#{next(self._counter)}",
            shape=array.shape,
            dtype=array.dtype,
            read=lambda region, array=array: array[region.slices],
        )
    )

  def run(self, jaxpr: Any, args: Sequence[VirtualLeaf]) -> list[VirtualLeaf]:
    """Interprets `jaxpr` with `args` bound to its inputs.

    Args:
      jaxpr: A closed jaxpr, or a jaxpr carrying its own constants.
      args: One symbolic value per input.

    Returns:
      One symbolic value per output.
    """
    inner = getattr(jaxpr, "jaxpr", jaxpr)
    env: dict[Any, VirtualLeaf] = {}
    for var, const in zip(inner.constvars, getattr(jaxpr, "consts", ())):
      env[var] = self._const(const)
    env.update(zip(inner.invars, args))

    def read(atom):
      if isinstance(atom, jax_core.Literal):
        return self._const(np.asarray(atom.val, dtype=atom.aval.dtype))
      return env[atom]

    for eqn in inner.eqns:
      operands = [read(a) for a in eqn.invars]
      avals = [v.aval for v in eqn.outvars]
      try:
        outs = self._apply(eqn, operands, avals)
      except (rules_lib.Unsupported, rules_lib.TooManyPieces):
        outs = self._derive(eqn, operands, avals)
      for var, out in zip(eqn.outvars, outs):
        if not isinstance(var, jax_core.DropVar):
          env[var] = out
    return [read(a) for a in inner.outvars]

  def _apply(
      self, eqn: Any, operands: list[VirtualLeaf], avals: list[Any]
  ) -> list[VirtualLeaf]:
    """Applies the rule for one equation.

    Args:
      eqn: The jaxpr equation.
      operands: Its operands as symbolic values.
      avals: The abstract values of its results.

    Returns:
      One symbolic value per result.

    Raises:
      Unsupported: When no rule covers the primitive or its operands.
      TooManyPieces: When the rule would split a value too finely.
    """
    name = eqn.primitive.name
    params = eqn.params
    shape = tuple(avals[0].shape) if avals else ()
    dtype = np.dtype(avals[0].dtype) if avals else None
    if name in self._rules.calls:
      sub = next(
          (params[p] for p in _SUBJAXPR_PARAMS if params.get(p) is not None),
          None,
      )
      if sub is None:
        raise rules_lib.Unsupported()
      return self.run(sub, operands)
    rule = self._rules.layout.get(name)
    if rule is not None:
      return rule(operands, params, shape, dtype)
    if all(isinstance(a, jax_core.Literal) for a in eqn.invars):
      args = [np.asarray(a.val, dtype=a.aval.dtype) for a in eqn.invars]
      outs = host.bind_on_cpu(eqn.primitive, params, *args)
      return [self._const(o) for o in outs]
    if name in self._rules.elementwise:
      return [
          rules_lib.elementwise(
              eqn.primitive, dict(params), operands, shape, dtype
          )
      ]
    raise rules_lib.Unsupported()

  def _derive(
      self, eqn: Any, operands: list[VirtualLeaf], avals: list[Any]
  ) -> list[VirtualLeaf]:
    """Computes an equation whole, exposing each result as a derived source."""
    primitive, params = eqn.primitive, eqn.params
    return fill.derive(
        f"{primitive.name}#{next(self._counter)}",
        operands,
        functools.partial(host.bind_on_cpu, primitive, params),
        [(a.shape, a.dtype) for a in avals],
    )


def _materialized_by(value: VirtualLeaf) -> list[str]:
  """Names the primitives whose full inputs must be read to build `value`."""
  names, seen = set(), set()
  pending = [ref for a in value.assignments for ref in a.refs()]
  while pending:
    ref = pending.pop()
    if id(ref) in seen or not isinstance(ref, DerivedRef):
      continue
    seen.add(id(ref))
    if ref.inputs:
      names.add(ref.key.split("#", 1)[0])
    pending.extend(ref.inputs)
  return sorted(names)


def _make_read(spec: Any) -> boxes.ReadFn | None:
  """Returns a sub-range reader for a source leaf, or `None` for metadata."""
  if isinstance(spec, boxes.StoredArray):
    return spec.read
  if isinstance(spec, (np.ndarray, jax.Array)):
    host = functools.cache(lambda: np.asarray(spec))
    return lambda region: host()[region.slices]
  return None


@dataclasses.dataclass
class ResolvedPlan:
  """A surgery function resolved against a source, before any data moves.

  Attributes:
    manifest: One virtual leaf per output key.
    report: What the plan does, including any errors.
  """

  manifest: boxes.Manifest
  report: report_lib.PlanReport


def resolve(
    fn: Callable[[Any], Any],
    source: Any,
    target: Any = None,
    *,
    rules: rules_lib.Rules = rules_lib.DEFAULT_RULES,
) -> ResolvedPlan:
  """Traces `fn` against `source` and returns the boxes it implies.

  Args:
    fn: Maps the source tree to the target tree with ordinary JAX operations.
    source: The source tree: arrays or stored arrays to execute, metadata or
      abstract leaves for a preview. Only shapes and dtypes are traced.
    target: The abstract target tree, or `None` to skip conformance checks.
    rules: How each primitive becomes boxes.

  Returns:
    The manifest and its report.
  """
  refs = [
      LeafRef(
          SOURCE, key, tuple(spec.shape), np.dtype(spec.dtype), _make_read(spec)
      )
      for key, spec in tree_utils.to_flat_dict(source, sep=".").items()
  ]
  abstract = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(source),
      [jax.ShapeDtypeStruct(r.shape, r.dtype) for r in refs],
  )
  jaxpr, out_shapes = jax.make_jaxpr(fn, return_shape=True)(abstract)
  outs = _Interpreter(rules).run(jaxpr, [boxes.as_virtual(r) for r in refs])
  report = report_lib.PlanReport()
  manifest = dict(zip(tree_utils.to_flat_dict(out_shapes, sep="."), outs))
  for key, leaf in manifest.items():
    names = _materialized_by(leaf)
    if names:
      report.materialized[key] = names
  if target is not None:
    _check_target(manifest, tree_utils.to_flat_dict(target, sep="."), report)
  return ResolvedPlan(manifest, report)


def _check_target(
    manifest: boxes.Manifest,
    target: dict[str, Any],
    report: report_lib.PlanReport,
) -> None:
  """Records every way `manifest` fails to produce exactly `target`."""
  for key, spec in target.items():
    leaf = manifest.get(key)
    if leaf is None:
      report.add_error(
          report_lib.MISSING_KEY,
          key,
          f"{key!r} is not produced by the function",
      )
      continue
    if tuple(leaf.shape) != tuple(spec.shape):
      report.add_error(
          report_lib.SHAPE_MISMATCH,
          key,
          f"produced shape {tuple(leaf.shape)} != target {tuple(spec.shape)}",
      )
    if np.dtype(leaf.dtype) != np.dtype(spec.dtype):
      report.add_error(
          report_lib.DTYPE_MISMATCH,
          key,
          f"produced dtype {np.dtype(leaf.dtype)} != target"
          f" {np.dtype(spec.dtype)}",
      )
  for key in manifest:
    if key not in target:
      report.add_error(
          report_lib.UNEXPECTED_KEY,
          key,
          f"{key!r} is produced but has no place in the target",
      )


def preview(
    fn: Callable[[Any], Any],
    source: Any,
    target: Any = None,
    *,
    rules: rules_lib.Rules = rules_lib.DEFAULT_RULES,
) -> report_lib.PlanReport:
  """Traces `fn` and reports what loading through it would do, reading nothing.

  Args:
    fn: Maps the source tree to the target tree with ordinary JAX operations.
    source: Source metadata or tree.
    target: The abstract target tree, or `None`.
    rules: How each primitive becomes boxes.

  Returns:
    The plan report: errors, and which outputs need whole-array computation.
  """
  return resolve(fn, source, target, rules=rules).report


def read_keys(
    fn: Callable[[Any], Any],
    source_metadata: Any,
    *,
    rules: rules_lib.Rules = rules_lib.DEFAULT_RULES,
) -> set[str]:
  """Returns the dotted source keys loading through `fn` will read.

  Resolves against metadata alone, so it moves no array data.

  Args:
    fn: The surgery function.
    source_metadata: Source metadata, for example from `ocp.metadata`.
    rules: How each primitive becomes boxes.

  Returns:
    The keys whose bytes execution reads; every other key is never opened.
  """
  manifest = resolve(fn, source_metadata, rules=rules).manifest
  return {
      ref.key
      for leaf in manifest.values()
      for ref in boxes.stored_refs(leaf)
      if ref.source == SOURCE
  }
