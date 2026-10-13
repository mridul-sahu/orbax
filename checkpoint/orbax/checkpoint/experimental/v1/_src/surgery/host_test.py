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


"""Tests for the host backend."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import boxes
from orbax.checkpoint.experimental.v1._src.surgery import fill
from orbax.checkpoint.experimental.v1._src.surgery import host

Region = boxes.Region
LeafRef = boxes.LeafRef
VirtualLeaf = boxes.VirtualLeaf


class HostBackendTest(parameterized.TestCase):

  def _ref(self, array):
    return LeafRef(
        source="source",
        key="w",
        shape=array.shape,
        dtype=array.dtype,
        read=lambda region: array[region.slices],
    )

  @parameterized.parameters(
      (np.float32, jnp.bfloat16),
      (np.float32, np.float16),
      (jnp.bfloat16, np.float32),
      (np.float16, jnp.bfloat16),
  )
  def test_copy_cast_rounds_like_xla(self, src, dst):
    rng = np.random.default_rng(0)
    values = rng.standard_normal(4096) * np.exp2(rng.integers(-30, 30, 4096))
    array = np.asarray(jnp.asarray(values, jnp.float32).astype(src))
    leaf = boxes.as_virtual(self._ref(array))
    eqn = jax.make_jaxpr(lambda x: x.astype(dst))(array).eqns[0]
    cast = boxes.Assignment(
        leaf.assignments[0].target_region,
        boxes.Apply(eqn.primitive, eqn.params, (leaf.assignments[0].expr,)),
    )
    got = fill.assemble_host(VirtualLeaf(leaf.shape, np.dtype(dst), (cast,)))
    self.assertEqual(
        got.tobytes(), np.asarray(jnp.asarray(array).astype(dst)).tobytes()
    )

  @parameterized.parameters((300, 600), (512, 256), (1000, 3))
  def test_tiled_copy_transposes_exactly(self, rows, cols):
    src = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    out = np.empty((cols, rows), np.float32)
    host.copy_tiled(out, src.T)
    np.testing.assert_array_equal(out, src.T)


if __name__ == "__main__":
  absltest.main()
