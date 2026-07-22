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

"""Tests for save-side surgery and the fine-tune-then-export round trip."""

import os

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.v1 import surgery
import orbax.checkpoint.experimental.v1 as ocp


def _sds(shape, dtype):
  return jax.ShapeDtypeStruct(tuple(shape), np.dtype(dtype))


class SaveTest(absltest.TestCase):

  def test_fine_tune_then_export_round_trip(self):
    root = self.create_tempdir().full_path
    in_path = os.path.join(root, "in")
    out_path = os.path.join(root, "out")

    original = {
        "layers.0.w": jnp.array([1, 2], dtype=jnp.float32),
        "layers.1.w": jnp.array([3, 4], dtype=jnp.float32),
    }
    ocp.save(in_path, original)

    import_plan = surgery.pipeline(surgery.stack(r"layers\.(\d+\.)", axis=0))
    imported = surgery.load(
        in_path, import_plan, target={"layers.w": _sds((2, 2), jnp.float32)}
    )

    fine_tuned = jax.tree.map(lambda x: x + 10, imported)

    surgery.save(out_path, fine_tuned, import_plan.inverse())

    reloaded = ocp.load(out_path)
    self.assertEqual(set(reloaded), {"layers.0.w", "layers.1.w"})
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(reloaded["layers.0.w"])),
        np.array([11, 12], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(reloaded["layers.1.w"])),
        np.array([13, 14], dtype=np.float32),
    )


if __name__ == "__main__":
  absltest.main()
