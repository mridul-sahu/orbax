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


"""Tests for the device backend's layout of fetched blocks."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import device
from orbax.checkpoint.experimental.v1._src.surgery import host


class RelayoutTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("transpose", (6, 4), (0, 0), ((0, 1), (1, 0)), (4, 6)),
      ("reversed_strided", (9, 4), (8, 3), ((-2, 0), (0, -1)), (5, 4)),
      ("split_axis", (24,), (0,), ((8, 1),), (3, 8)),
      ("reordered_split", (24,), (0,), ((1, 3),), (3, 8)),
      ("broadcast", (4,), (0,), ((0, 1),), (3, 4)),
      ("diagonal_gather", (5, 5), (0, 0), ((1,), (1,)), (5,)),
  )
  def test_matches_host_strided_view(self, block_shape, offset, lattice, shape):
    block = np.arange(np.prod(block_shape), dtype=np.float32).reshape(
        block_shape
    )
    want = host.strided(
        block, np.asarray(offset), np.asarray(lattice, dtype=np.int64), shape
    )
    got = device._relayout_fn(offset, lattice, shape)(jax.device_put(block))
    np.testing.assert_array_equal(np.asarray(got), want)


if __name__ == "__main__":
  absltest.main()
