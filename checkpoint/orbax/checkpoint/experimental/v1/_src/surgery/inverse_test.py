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

"""Tests for plan inversion and the import/export round trip."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.v1 import surgery


def _sds(shape, dtype):
  return jax.ShapeDtypeStruct(tuple(shape), np.dtype(dtype))


class InverseTest(absltest.TestCase):

  def test_import_then_inverse_round_trips_keys_and_shapes(self):
    source = {
        "model.layers.0.gate.w": jnp.arange(4, dtype=jnp.float32),
        "model.layers.0.up.w": jnp.arange(4, dtype=jnp.float32) + 4,
        "model.layers.1.gate.w": jnp.arange(4, dtype=jnp.float32) + 8,
        "model.layers.1.up.w": jnp.arange(4, dtype=jnp.float32) + 12,
    }
    plan = surgery.pipeline(
        surgery.rename((r"^model\.", "")),
        surgery.fuse(parts=("gate.w", "up.w"), into="gate_up.w", axis=0),
        surgery.stack(r"layers\.(\d+\.)", axis=0),
    )
    imported_target = {"layers.gate_up.w": _sds((2, 8), jnp.float32)}
    imported = surgery.load(source, plan, target=imported_target)

    export_plan = plan.inverse()
    original_target = {k: _sds(v.shape, v.dtype) for k, v in source.items()}
    restored = surgery.load(imported, export_plan, target=original_target)

    self.assertEqual(set(restored), set(source))
    for key, value in source.items():
      np.testing.assert_array_equal(
          np.asarray(jax.device_get(restored[key])),
          np.asarray(jax.device_get(value)),
      )

  def test_cast_inverse_restores_dtype_with_a_note(self):
    source = {"w": jnp.arange(4, dtype=jnp.float32)}
    plan = surgery.pipeline(surgery.cast(r".*", jnp.float16))
    imported = surgery.load(source, plan, target={"w": _sds((4,), jnp.float16)})
    self.assertEqual(imported["w"].dtype, jnp.float16)

    export_plan = plan.inverse()
    original_target = {"w": _sds((4,), jnp.float32)}
    report = export_plan.preview(imported, original_target)
    self.assertEmpty(report.errors)
    self.assertIn("w", report.filled)
    self.assertIn("lossy", report.filled["w"])

    restored = surgery.load(imported, export_plan, target=original_target)
    self.assertEqual(restored["w"].dtype, jnp.float32)

  def test_non_invertible_op_raises(self):
    plan = surgery.pipeline(surgery.drop(r"^opt\."))
    with self.assertRaises(surgery.SurgeryError):
      plan.inverse()


if __name__ == "__main__":
  absltest.main()
