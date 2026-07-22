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

"""Tests for the manifest effect of each surgery operation."""

from absl.testing import absltest
import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import manifest as manifest_lib
from orbax.checkpoint.experimental.v1._src.surgery import operations as ops
from orbax.checkpoint.experimental.v1._src.surgery import report as report_lib

LeafRef = manifest_lib.LeafRef
VirtualLeaf = manifest_lib.VirtualLeaf


def _ref(key, shape, dtype="float32"):
  return LeafRef(
      source="source", key=key, shape=tuple(shape), dtype=np.dtype(dtype)
  )


def _run(op, manifest, on_missing="error", source_refs=None):
  ctx = ops.ResolveContext(
      on_missing=on_missing,
      report=report_lib.PlanReport(),
      source_refs=source_refs or {},
  )
  return op(manifest, ctx), ctx.report


def _kinds(report):
  return [e.kind for e in report.errors]


def _zeros_init(shape, dtype):
  return np.zeros(shape, dtype)


class RenameTest(absltest.TestCase):

  def test_renames_and_records_moves(self):
    manifest = {
        "model.a": _ref("model.a", (2,)),
        "model.b": _ref("model.b", (2,)),
    }
    result, report = _run(ops.rename((r"^model\.", "")), manifest)
    self.assertEqual(set(result), {"a", "b"})
    self.assertCountEqual(report.moved, [("model.a", "a"), ("model.b", "b")])
    self.assertEmpty(report.errors)

  def test_unmatched_rule_is_error(self):
    manifest = {"a": _ref("a", (2,))}
    _, report = _run(ops.rename((r"^zzz", "x")), manifest)
    self.assertEqual(_kinds(report), [report_lib.UNMATCHED_RULE])

  def test_collision_is_error(self):
    manifest = {"a.x": _ref("a.x", (2,)), "b.x": _ref("b.x", (2,))}
    _, report = _run(ops.rename((r"^[ab]\.", "c.")), manifest)
    self.assertIn(report_lib.RENAME_COLLISION, _kinds(report))


class DropSelectTest(absltest.TestCase):

  def test_drop_removes_and_records(self):
    manifest = {"w": _ref("w", (2,)), "opt": _ref("opt", (2,))}
    result, report = _run(ops.drop(r"^opt$"), manifest)
    self.assertEqual(set(result), {"w"})
    self.assertEqual(report.dropped, ["opt"])

  def test_select_keeps_only_matching(self):
    manifest = {"w": _ref("w", (2,)), "opt": _ref("opt", (2,))}
    result, report = _run(ops.select(r"^w$"), manifest)
    self.assertEqual(set(result), {"w"})
    self.assertEqual(report.dropped, ["opt"])

  def test_select_matching_nothing_is_error(self):
    manifest = {"w": _ref("w", (2,))}
    _, report = _run(ops.select(r"^zzz$"), manifest)
    self.assertEqual(_kinds(report), [report_lib.UNMATCHED_RULE])


class StackTest(absltest.TestCase):

  def test_stacks_indexed_keys(self):
    manifest = {
        "layers.0.w": _ref("layers.0.w", (2, 3)),
        "layers.1.w": _ref("layers.1.w", (2, 3)),
        "other": _ref("other", (5,)),
    }
    result, report = _run(ops.stack(r"layers\.(\d+\.)", axis=0), manifest)
    self.assertEqual(set(result), {"layers.w", "other"})
    leaf = result["layers.w"]
    self.assertEqual(leaf.shape, (2, 2, 3))
    self.assertLen(leaf.assignments, 2)
    self.assertEqual({a.inserted_axis for a in leaf.assignments}, {0})
    self.assertEqual(
        report.stacked["layers.w"], {0: "layers.0.w", 1: "layers.1.w"}
    )

  def test_gap_without_filler_is_error(self):
    manifest = {
        "layers.0.w": _ref("layers.0.w", (2,)),
        "layers.2.w": _ref("layers.2.w", (2,)),
    }
    _, report = _run(ops.stack(r"layers\.(\d+\.)", axis=0), manifest)
    self.assertIn(report_lib.INCOMPLETE_GROUP, _kinds(report))

  def test_gap_with_filler_fills(self):
    manifest = {
        "layers.0.w": _ref("layers.0.w", (2,)),
        "layers.2.w": _ref("layers.2.w", (2,)),
    }
    result, report = _run(
        ops.stack(r"layers\.(\d+\.)", axis=0, filler=0.0), manifest
    )
    self.assertEmpty(report.errors)
    self.assertEqual(result["layers.w"].shape, (3, 2))
    self.assertIsNotNone(result["layers.w"].init)
    self.assertIn("layers.w", report.filled)

  def test_no_match_is_error(self):
    manifest = {"w": _ref("w", (2,))}
    _, report = _run(ops.stack(r"layers\.(\d+\.)", axis=0), manifest)
    self.assertEqual(_kinds(report), [report_lib.UNMATCHED_RULE])


class UnstackTest(absltest.TestCase):

  def test_unstacks_leading_axis(self):
    manifest = {"s": _ref("s", (3, 2))}
    result, report = _run(ops.unstack(r"^s$", axis=0), manifest)
    self.assertEqual(set(result), {"s.0", "s.1", "s.2"})
    self.assertEqual(result["s.0"].shape, (2,))
    self.assertEqual(result["s.0"].assignments[0].sliced_source_axis, 0)
    self.assertEmpty(report.errors)


class FuseSplitTest(absltest.TestCase):

  def test_fuse_concatenates_parts(self):
    manifest = {
        "l.gate.w": _ref("l.gate.w", (4,)),
        "l.up.w": _ref("l.up.w", (4,)),
    }
    result, report = _run(
        ops.fuse(parts=("gate.w", "up.w"), into="gate_up.w", axis=0), manifest
    )
    self.assertEqual(set(result), {"l.gate_up.w"})
    self.assertEqual(result["l.gate_up.w"].shape, (8,))
    self.assertEqual(
        report.fused["l.gate_up.w"], ["l.gate.w", "l.up.w"]
    )

  def test_fuse_incomplete_group_is_error(self):
    manifest = {"l.gate.w": _ref("l.gate.w", (4,))}
    _, report = _run(
        ops.fuse(parts=("gate.w", "up.w"), into="gate_up.w", axis=0), manifest
    )
    self.assertIn(report_lib.INCOMPLETE_GROUP, _kinds(report))

  def test_split_slices_into_parts(self):
    manifest = {"qkv": _ref("qkv", (6, 4))}
    result, report = _run(
        ops.split(key="qkv", sizes=(2, 2, 2), into=("q", "k", "v"), axis=0),
        manifest,
    )
    self.assertEqual(set(result), {"q", "k", "v"})
    self.assertEqual(result["q"].shape, (2, 4))
    self.assertEmpty(report.errors)

  def test_split_size_mismatch_is_error(self):
    manifest = {"qkv": _ref("qkv", (6, 4))}
    _, report = _run(
        ops.split(key="qkv", sizes=(2, 2), into=("q", "k"), axis=0), manifest
    )
    self.assertIn(report_lib.SHAPE_MISMATCH, _kinds(report))


class RepeatCastResizeTest(absltest.TestCase):

  def test_repeat_tiles_source(self):
    manifest = {"kv": _ref("kv", (2, 4))}
    result, _ = _run(ops.repeat(r"^kv$", axis=0, times=3), manifest)
    self.assertEqual(result["kv"].shape, (6, 4))
    self.assertLen(result["kv"].assignments, 3)

  def test_cast_changes_dtype_and_annotates(self):
    manifest = {"w": _ref("w", (2,), "float32")}
    result, _ = _run(ops.cast(r".*", "float16"), manifest)
    self.assertEqual(result["w"].dtype, np.dtype("float16"))
    self.assertEqual(result["w"].assignments[0].cast, np.dtype("float16"))

  def test_resize_truncates(self):
    manifest = {"emb": _ref("emb", (10, 4))}
    result, report = _run(ops.resize(r"^emb$", axis=0, size=6), manifest)
    self.assertEqual(result["emb"].shape, (6, 4))
    self.assertIsNone(result["emb"].init)
    self.assertEmpty(report.errors)

  def test_resize_grows_with_init(self):
    manifest = {"emb": _ref("emb", (10, 4))}
    result, report = _run(
        ops.resize(r"^emb$", axis=0, size=16, init=_zeros_init), manifest
    )
    self.assertEqual(result["emb"].shape, (16, 4))
    self.assertIsNotNone(result["emb"].init)
    self.assertIn("emb", report.filled)

  def test_resize_grow_without_init_is_error(self):
    manifest = {"emb": _ref("emb", (10, 4))}
    _, report = _run(ops.resize(r"^emb$", axis=0, size=16), manifest)
    self.assertIn(report_lib.UNCOVERED_TARGET, _kinds(report))


class TakeTest(absltest.TestCase):

  def test_imports_from_named_source(self):
    refs = {
        "vision": {
            "params.a": _ref("params.a", (2,)),
            "params.b": _ref("params.b", (2,)),
        }
    }
    result, report = _run(
        ops.take("vision", r"^params\.", into="img."),
        {},
        source_refs=refs,
    )
    self.assertEqual(set(result), {"img.a", "img.b"})
    self.assertEmpty(report.errors)

  def test_missing_namespace_is_error(self):
    _, report = _run(
        ops.take("vision", r"^params\.", into="img."), {}, source_refs={}
    )
    self.assertEqual(_kinds(report), [report_lib.MISSING_SOURCE])


class ErrorCollectionTest(absltest.TestCase):

  def test_multiple_defects_accumulate_in_one_call(self):
    manifest = {"a": _ref("a", (2,))}
    _, report = _run(
        ops.rename((r"^x", "1"), (r"^y", "2")), manifest
    )
    self.assertLen(report.errors, 2)


if __name__ == "__main__":
  absltest.main()
