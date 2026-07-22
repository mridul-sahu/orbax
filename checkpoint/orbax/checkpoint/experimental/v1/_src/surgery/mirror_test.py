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

"""Tests for mirroring structural ops onto optimizer subtrees."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.v1 import surgery
from orbax.checkpoint.experimental.v1._src.surgery import report as report_lib


def _sds(shape, dtype):
  return jax.ShapeDtypeStruct(tuple(shape), np.dtype(dtype))


def _moment_source():
  return {
      "layers.0.w": jnp.array([1, 2], dtype=jnp.float32),
      "layers.1.w": jnp.array([3, 4], dtype=jnp.float32),
      "opt.mu.layers.0.w": jnp.array([10, 20], dtype=jnp.float32),
      "opt.mu.layers.1.w": jnp.array([30, 40], dtype=jnp.float32),
      "opt.nu.layers.0.w": jnp.array([100, 200], dtype=jnp.float32),
      "opt.nu.layers.1.w": jnp.array([300, 400], dtype=jnp.float32),
  }


class MirrorTest(absltest.TestCase):

  def test_mirror_reshapes_parameter_and_moments_together(self):
    plan = surgery.pipeline(
        surgery.stack(r"^layers\.(\d+\.)", axis=0),
        surgery.mirror(onto=("opt.mu.", "opt.nu.")),
    )
    target = {
        "layers.w": _sds((2, 2), jnp.float32),
        "opt.mu.layers.w": _sds((2, 2), jnp.float32),
        "opt.nu.layers.w": _sds((2, 2), jnp.float32),
    }

    report = plan.preview(_moment_source(), target)
    self.assertEmpty(report.errors)

    result = surgery.load(_moment_source(), plan, target=target)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["layers.w"])),
        np.array([[1, 2], [3, 4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["opt.mu.layers.w"])),
        np.array([[10, 20], [30, 40]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["opt.nu.layers.w"])),
        np.array([[100, 200], [300, 400]], dtype=np.float32),
    )

  def test_uncovered_moment_is_orphaned(self):
    plan = surgery.pipeline(
        surgery.stack(r"^layers\.(\d+\.)", axis=0),
        surgery.mirror(onto=("opt.mu.",)),
    )
    report = plan.preview(_moment_source())
    kinds = [e.kind for e in report.errors]
    self.assertIn(report_lib.ORPHANED_OPTIMIZER_STATE, kinds)
    orphans = {
        e.key
        for e in report.errors
        if e.kind == report_lib.ORPHANED_OPTIMIZER_STATE
    }
    self.assertEqual(
        orphans, {"opt.nu.layers.0.w", "opt.nu.layers.1.w"}
    )


if __name__ == "__main__":
  absltest.main()
