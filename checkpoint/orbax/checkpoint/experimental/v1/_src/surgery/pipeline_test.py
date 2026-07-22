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

"""Tests for plan resolution, previews, and validation."""

from absl.testing import absltest
import jax
import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import manifest as manifest_lib
from orbax.checkpoint.experimental.v1._src.surgery import operations as ops
from orbax.checkpoint.experimental.v1._src.surgery import pipeline as pipeline_lib
from orbax.checkpoint.experimental.v1._src.surgery import report as report_lib

Region = manifest_lib.Region
Assignment = manifest_lib.Assignment
VirtualLeaf = manifest_lib.VirtualLeaf
LeafRef = manifest_lib.LeafRef


def _sds(shape, dtype="float32"):
  return jax.ShapeDtypeStruct(tuple(shape), np.dtype(dtype))


def _kinds(report):
  return [e.kind for e in report.errors]


class ResolveTest(absltest.TestCase):

  def test_rename_produces_renamed_manifest(self):
    source = {"model.a": _sds((2,)), "model.b": _sds((2,))}
    plan = pipeline_lib.pipeline(ops.rename((r"^model\.", "")))
    resolved = plan.resolve(source)
    self.assertEqual(set(resolved.manifest), {"a", "b"})
    self.assertCountEqual(
        resolved.report.moved, [("model.a", "a"), ("model.b", "b")]
    )

  def test_preview_matches_target(self):
    source = {
        "layers.0.w": _sds((2, 3)),
        "layers.1.w": _sds((2, 3)),
    }
    target = {"layers.w": _sds((2, 2, 3))}
    plan = pipeline_lib.pipeline(ops.stack(r"layers\.(\d+\.)", axis=0))
    report = plan.preview(source, target)
    self.assertEmpty(report.errors)
    self.assertIn("layers.w", report.stacked)

  def test_shape_mismatch_against_target(self):
    source = {"w": _sds((2, 3))}
    target = {"w": _sds((2, 4))}
    report = pipeline_lib.pipeline().preview(source, target)
    self.assertEqual(_kinds(report), [report_lib.SHAPE_MISMATCH])

  def test_dtype_mismatch_against_target(self):
    source = {"w": _sds((2,), "float32")}
    target = {"w": _sds((2,), "float16")}
    report = pipeline_lib.pipeline().preview(source, target)
    self.assertEqual(_kinds(report), [report_lib.DTYPE_MISMATCH])

  def test_unexpected_key_against_target(self):
    source = {"a": _sds((2,)), "b": _sds((2,))}
    target = {"a": _sds((2,))}
    report = pipeline_lib.pipeline().preview(source, target)
    self.assertEqual(_kinds(report), [report_lib.UNEXPECTED_KEY])


class MissingTargetPolicyTest(absltest.TestCase):

  def test_on_missing_error(self):
    source = {"a": _sds((2,))}
    target = {"a": _sds((2,)), "fresh": _sds((3,))}
    report = pipeline_lib.pipeline(on_missing="error").preview(source, target)
    self.assertEqual(_kinds(report), [report_lib.MISSING_SOURCE])

  def test_on_missing_init_from_target(self):
    source = {"a": _sds((2,))}
    target = {"a": _sds((2,)), "fresh": _sds((3,))}
    plan = pipeline_lib.pipeline(on_missing="init_from_target")
    resolved = plan.resolve(source, target)
    self.assertEmpty(resolved.report.errors)
    self.assertEqual(resolved.report.filled["fresh"], "init_from_target")
    self.assertIn("fresh", resolved.manifest)

  def test_invalid_on_missing_raises(self):
    with self.assertRaises(ValueError):
      pipeline_lib.pipeline(on_missing="pad")


class CoverageValidationTest(absltest.TestCase):

  def _ctx(self):
    return ops.ResolveContext(
        on_missing="error", report=report_lib.PlanReport()
    )

  def _ref(self):
    return LeafRef(
        source="source", key="w", shape=(4,), dtype=np.dtype("float32")
    )

  def _slab(self, lo, hi):
    return Assignment(
        target_region=Region((lo,), (hi,)),
        ref=self._ref(),
        source_region=Region((0,), (hi - lo,)),
    )

  def test_gap_without_init_is_uncovered(self):
    manifest = {
        "w": VirtualLeaf(
            shape=(4,),
            dtype=np.dtype("float32"),
            assignments=(self._slab(0, 2),),
        )
    }
    ctx = self._ctx()
    pipeline_lib._validate(manifest, None, ctx)
    self.assertEqual(_kinds(ctx.report), [report_lib.UNCOVERED_TARGET])

  def test_overlap_is_error(self):
    manifest = {
        "w": VirtualLeaf(
            shape=(4,),
            dtype=np.dtype("float32"),
            assignments=(self._slab(0, 3), self._slab(2, 4)),
        )
    }
    ctx = self._ctx()
    pipeline_lib._validate(manifest, None, ctx)
    self.assertIn(report_lib.OVERLAPPING_ASSIGNMENTS, _kinds(ctx.report))

  def test_full_cover_passes(self):
    manifest = {
        "w": VirtualLeaf(
            shape=(4,),
            dtype=np.dtype("float32"),
            assignments=(self._slab(0, 2), self._slab(2, 4)),
        )
    }
    ctx = self._ctx()
    pipeline_lib._validate(manifest, None, ctx)
    self.assertEmpty(ctx.report.errors)


class ErrorCollectionTest(absltest.TestCase):

  def test_all_errors_collected_in_one_preview(self):
    source = {"w": _sds((2, 3))}
    target = {"w": _sds((2, 4))}
    plan = pipeline_lib.pipeline(ops.rename((r"^zzz", "x")))
    report = plan.preview(source, target)
    self.assertCountEqual(
        _kinds(report),
        [report_lib.UNMATCHED_RULE, report_lib.SHAPE_MISMATCH],
    )

  def test_raise_if_errors(self):
    source = {"a": _sds((2,))}
    target = {"missing": _sds((2,))}
    report = pipeline_lib.pipeline().preview(source, target)
    with self.assertRaises(report_lib.SurgeryError):
      report.raise_if_errors()


if __name__ == "__main__":
  absltest.main()
