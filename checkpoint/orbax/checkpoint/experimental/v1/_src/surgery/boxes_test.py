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


"""Tests for regions and the box model."""

from absl.testing import absltest
from orbax.checkpoint.experimental.v1._src.surgery import boxes

Region = boxes.Region


class RegionTest(absltest.TestCase):

  def test_full_and_properties(self):
    region = Region.full((2, 3, 4))
    self.assertEqual(region.start, (0, 0, 0))
    self.assertEqual(region.stop, (2, 3, 4))
    self.assertEqual(region.shape, (2, 3, 4))
    self.assertEqual(region.size, 24)

  def test_intersect(self):
    a = Region((0, 0), (4, 4))
    self.assertEqual(
        a.intersect(Region((2, 1), (6, 3))), Region((2, 1), (4, 3))
    )
    self.assertIsNone(a.intersect(Region((4, 4), (6, 6))))

  def test_shifted(self):
    self.assertEqual(
        Region((0, 5), (3, 9)).shifted((2, -5)), Region((2, 0), (5, 4))
    )


if __name__ == "__main__":
  absltest.main()
