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

"""Tests for the compute escape hatch."""

import os

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.v1 import surgery
import orbax.checkpoint.experimental.v1 as ocp


def _sds(shape, dtype):
  return jax.ShapeDtypeStruct(tuple(shape), np.dtype(dtype))


class ComputeTest(absltest.TestCase):

  def test_compute_runs_the_function(self):
    source = {"w": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)}
    plan = surgery.pipeline(surgery.compute(r"^w$", lambda a: a * 2))
    result = surgery.load(source, plan, target={"w": _sds((2, 3), jnp.float32)})
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["w"])),
        np.arange(6, dtype=np.float32).reshape(2, 3) * 2,
    )

  def test_compute_leaves_dropped_key_unread(self):
    directory = os.path.join(self.create_tempdir().full_path, "ckpt")
    ocp.save(
        directory,
        {
            "w": jnp.arange(4, dtype=jnp.float32),
            "dropped": jnp.arange(4, dtype=jnp.float32) + 100,
        },
    )
    plan = surgery.pipeline(
        surgery.drop(r"^dropped$"),
        surgery.compute(r"^w$", lambda a: a + 1),
    )

    read = surgery.read_keys(plan, ocp.metadata(directory).metadata)
    self.assertIn("w", read)
    self.assertNotIn("dropped", read)

    result = surgery.load(
        directory, plan, target={"w": _sds((4,), jnp.float32)}
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["w"])),
        np.arange(4, dtype=np.float32) + 1,
    )


if __name__ == "__main__":
  absltest.main()
