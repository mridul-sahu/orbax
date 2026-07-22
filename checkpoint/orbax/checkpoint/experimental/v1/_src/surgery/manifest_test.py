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

"""Tests for the model surgery intermediate representation."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from orbax.checkpoint.experimental.v1._src.surgery import manifest as manifest_lib

Region = manifest_lib.Region
LeafRef = manifest_lib.LeafRef
VirtualLeaf = manifest_lib.VirtualLeaf
Assignment = manifest_lib.Assignment


def _ref(key, array):
  return LeafRef(
      source="source",
      key=key,
      shape=array.shape,
      dtype=array.dtype,
      read=lambda region, a=array: a[region.slices],
  )


class RegionTest(parameterized.TestCase):

  def test_full_and_properties(self):
    region = Region.full((2, 3, 4))
    self.assertEqual(region.start, (0, 0, 0))
    self.assertEqual(region.stop, (2, 3, 4))
    self.assertEqual(region.ndim, 3)
    self.assertEqual(region.shape, (2, 3, 4))
    self.assertEqual(region.size, 24)

  def test_slices_index_an_array(self):
    array = np.arange(24).reshape(2, 3, 4)
    region = Region((0, 1, 0), (2, 3, 2))
    np.testing.assert_array_equal(array[region.slices], array[0:2, 1:3, 0:2])

  def test_intersect_overlap(self):
    a = Region((0, 0), (4, 4))
    b = Region((2, 1), (6, 3))
    self.assertEqual(a.intersect(b), Region((2, 1), (4, 3)))

  def test_intersect_disjoint_is_none(self):
    a = Region((0, 0), (2, 2))
    b = Region((2, 2), (4, 4))
    self.assertIsNone(a.intersect(b))

  def test_compose_offsets_into_frame(self):
    outer = Region((10, 20), (14, 24))
    inner = Region((1, 1), (3, 2))
    self.assertEqual(outer.compose(inner), Region((11, 21), (13, 22)))

  def test_insert_and_remove_axis(self):
    region = Region((0, 5), (3, 9))
    inserted = region.with_inserted_axis(1, 7)
    self.assertEqual(inserted, Region((0, 7, 5), (3, 8, 9)))
    self.assertEqual(inserted.without_axis(1), region)


class VirtualViewTest(absltest.TestCase):

  def test_as_virtual_is_identity_for_virtual_leaf(self):
    leaf = VirtualLeaf(shape=(2,), dtype=np.dtype("float32"), assignments=())
    self.assertIs(manifest_lib.as_virtual(leaf), leaf)

  def test_as_virtual_wraps_ref_with_identity_assignment(self):
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    leaf = manifest_lib.as_virtual(_ref("w", array))
    self.assertEqual(leaf.shape, (2, 3))
    self.assertLen(leaf.assignments, 1)
    a = leaf.assignments[0]
    self.assertEqual(a.target_region, Region.full((2, 3)))
    self.assertIsNone(a.inserted_axis)


class ReadMappingTest(absltest.TestCase):

  def test_identity_read(self):
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    a = manifest_lib.as_virtual(_ref("w", array)).assignments[0]
    hit = Region((1, 0), (3, 2))
    src = manifest_lib.source_read_region(a, hit)
    self.assertEqual(src, Region((1, 0), (3, 2)))
    chunk = manifest_lib.align_to_target(a.ref.read(src), a)
    np.testing.assert_array_equal(chunk, array[1:3, 0:2])

  def test_stack_insert_read(self):
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    ref = _ref("expert", array)
    # Place the source at index 4 on a new leading axis.
    target_region = Region.full((2, 3)).with_inserted_axis(0, 4)
    a = Assignment(
        target_region=target_region,
        ref=ref,
        source_region=Region.full((2, 3)),
        inserted_axis=0,
    )
    hit = target_region  # a shard covering the whole placement
    src = manifest_lib.source_read_region(a, hit)
    self.assertEqual(src, Region.full((2, 3)))
    chunk = manifest_lib.align_to_target(a.ref.read(src), a)
    self.assertEqual(chunk.shape, (1, 2, 3))
    np.testing.assert_array_equal(chunk[0], array)

  def test_unstack_slice_read(self):
    array = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    ref = _ref("stacked", array)
    # Read index 2 off the leading axis into a rank-2 target.
    a = Assignment(
        target_region=Region.full((2, 3)),
        ref=ref,
        source_region=Region((2, 0, 0), (3, 2, 3)),
        sliced_source_axis=0,
    )
    hit = Region.full((2, 3))
    src = manifest_lib.source_read_region(a, hit)
    chunk = manifest_lib.align_to_target(a.ref.read(src), a)
    self.assertEqual(chunk.shape, (2, 3))
    np.testing.assert_array_equal(chunk, array[2])


class CoverageTest(absltest.TestCase):

  def _slab(self, ref, lo, hi):
    return Assignment(
        target_region=Region((lo,), (hi,)),
        ref=ref,
        source_region=Region((0,), (hi - lo,)),
    )

  def test_disjoint_slabs_have_no_overlap(self):
    ref = _ref("w", np.zeros(4, np.float32))
    assignments = (self._slab(ref, 0, 2), self._slab(ref, 2, 4))
    self.assertIsNone(manifest_lib.overlapping_assignments(assignments))
    self.assertEqual(manifest_lib.covered_size(assignments), 4)

  def test_overlapping_slabs_detected(self):
    ref = _ref("w", np.zeros(4, np.float32))
    assignments = (self._slab(ref, 0, 3), self._slab(ref, 2, 4))
    self.assertEqual(manifest_lib.overlapping_assignments(assignments), (0, 1))


if __name__ == "__main__":
  absltest.main()
