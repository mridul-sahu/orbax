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


"""Tests for read planning and read merging."""

from absl.testing import absltest
import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import boxes
from orbax.checkpoint.experimental.v1._src.surgery import reads

Region = boxes.Region


class ReadMergingTest(absltest.TestCase):

  def test_strided_region_shape_and_slices(self):
    region = Region((1, 0), (8, 4), (3, 1))
    self.assertEqual(region.shape, (3, 4))
    self.assertEqual(region.size, 12)
    np.testing.assert_array_equal(
        np.arange(32).reshape(8, 4)[region.slices],
        np.arange(32).reshape(8, 4)[1:8:3],
    )

  def test_adjacent_and_duplicate_reads_merge(self):
    rows = [Region((r, 0), (r + 2, 8)) for r in range(0, 32, 2)]
    merged = reads.merge_regions(rows + rows[:3], itemsize=4)
    self.assertEqual(merged, [Region((0, 0), (32, 8))])

  def test_distant_reads_stay_separate(self):
    far = 10_000_000
    merged = reads.merge_regions(
        [Region((0,), (1,)), Region((far,), (far + 1,))], itemsize=4
    )
    self.assertLen(merged, 2)

  def test_interleaved_lattices_merge_into_one_dense_read(self):
    merged = reads.merge_regions(
        [Region((0,), (8,), (2,)), Region((1,), (9,), (2,))], itemsize=4
    )
    self.assertEqual(merged, [Region((0,), (9,))])

  def test_sparse_lattices_stay_separate_when_dense_read_wastes(self):
    n = 4_000_000
    sparse = [Region((0,), (n,), (4,)), Region((1,), (n + 1,), (4,))]
    self.assertLen(reads.merge_regions(sparse, itemsize=4), 2)

  def test_single_strided_read_stays_strided(self):
    region = Region((0, 0), (16, 8), (2, 1))
    self.assertEqual(reads.merge_regions([region], itemsize=4), [region])

  def test_batch_issues_one_read_per_merged_block(self):
    calls = []
    data = np.arange(64, dtype=np.float32).reshape(16, 4)

    def read(region):
      calls.append(region)
      return data[region.slices]

    ref = boxes.LeafRef(
        source="s",
        key="w",
        shape=(16, 4),
        dtype=np.dtype(np.float32),
        read=read,
    )
    requests = [Region((r, 0), (r + 4, 4)) for r in (0, 4, 8, 12)]
    batch = reads.ReadBatch((ref, r) for r in requests)
    for r in requests:
      np.testing.assert_array_equal(batch.fetch(ref, r), data[r.slices])
    self.assertEqual(calls, [Region((0, 0), (16, 4))])


if __name__ == "__main__":
  absltest.main()
