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


"""Tests for filling shards and for derived sources."""

from absl.testing import absltest
import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import boxes
from orbax.checkpoint.experimental.v1._src.surgery import fill

Region = boxes.Region
LeafRef = boxes.LeafRef
VirtualLeaf = boxes.VirtualLeaf


class FillTest(absltest.TestCase):

  def _ref(self, array):
    return LeafRef(
        source="source",
        key="w",
        shape=array.shape,
        dtype=array.dtype,
        read=lambda region: array[region.slices],
    )

  def test_constant_and_view_boxes(self):
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    view = boxes.as_virtual(self._ref(array)).assignments[0]
    pad = boxes.Assignment(Region((2, 0), (3, 3)), boxes.Const(np.float32(-1)))
    leaf = VirtualLeaf((3, 3), np.dtype(np.float32), (view, pad))
    np.testing.assert_array_equal(
        fill.assemble_host(leaf),
        np.concatenate([array, np.full((1, 3), -1, np.float32)]),
    )

  def test_derived_source_runs_once_across_shards(self):
    calls = []

    def fn():
      calls.append(1)
      return [np.arange(8, dtype=np.float32)]

    leaf = fill.derive("init", [], fn, [((8,), np.float32)])[0]
    seen = {}
    fill.fill(
        leaf,
        {Region((i,), (i + 2,)): None for i in range(0, 8, 2)},
        lambda region, buf: seen.update({region.start[0]: buf}),
    )
    self.assertLen(calls, 1)
    np.testing.assert_array_equal(
        np.concatenate([seen[i] for i in range(0, 8, 2)]), np.arange(8)
    )


if __name__ == "__main__":
  absltest.main()
