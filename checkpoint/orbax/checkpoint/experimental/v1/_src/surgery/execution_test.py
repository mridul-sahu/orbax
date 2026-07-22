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

"""End-to-end tests for plan execution, in memory and from a checkpoint."""

import os

os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=2"
)

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.v1 import surgery
import orbax.checkpoint.experimental.v1 as ocp

Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
PartitionSpec = jax.sharding.PartitionSpec


def _sds(shape, dtype, sharding=None):
  return jax.ShapeDtypeStruct(tuple(shape), np.dtype(dtype), sharding=sharding)


class InMemoryExecutionTest(absltest.TestCase):

  def test_rename_fuse_stack_cast(self):
    source = {
        "model.layers.0.mlp.gate.w": jnp.array([1, 2], dtype=jnp.float32),
        "model.layers.0.mlp.up.w": jnp.array([3, 4], dtype=jnp.float32),
        "model.layers.1.mlp.gate.w": jnp.array([5, 6], dtype=jnp.float32),
        "model.layers.1.mlp.up.w": jnp.array([7, 8], dtype=jnp.float32),
    }
    plan = surgery.pipeline(
        surgery.rename((r"^model\.", "")),
        surgery.fuse(
            parts=("mlp.gate.w", "mlp.up.w"), into="mlp.gate_up.w", axis=0
        ),
        surgery.stack(r"layers\.(\d+\.)", axis=0),
        surgery.cast(r".*", jnp.float16),
    )
    target = {"layers.mlp.gate_up.w": _sds((2, 4), jnp.float16)}

    result = surgery.load(source, plan, target=target)

    self.assertEqual(set(result), {"layers.mlp.gate_up.w"})
    out = np.asarray(jax.device_get(result["layers.mlp.gate_up.w"]))
    self.assertEqual(out.dtype, np.dtype(jnp.float16))
    np.testing.assert_array_equal(
        out, np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float16)
    )


class CheckpointExecutionTest(absltest.TestCase):

  def test_dropped_key_is_not_read(self):
    directory = os.path.join(self.create_tempdir().full_path, "ckpt")
    ocp.save(
        directory,
        {
            "kept": jnp.arange(4, dtype=jnp.int32),
            "dropped": jnp.arange(4, dtype=jnp.int32) + 100,
        },
    )

    plan = surgery.pipeline(surgery.drop(r"^dropped$"))

    source_metadata = ocp.metadata(directory).metadata
    read = surgery.read_keys(plan, source_metadata)
    self.assertIn("kept", read)
    self.assertNotIn("dropped", read)

    target = {"kept": _sds((4,), jnp.int32)}
    result = surgery.load(directory, plan, target=target)

    self.assertEqual(set(result), {"kept"})
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["kept"])), np.arange(4)
    )


class CheckpointByteRangeTest(absltest.TestCase):

  def test_truncation_reads_only_the_kept_rows(self):
    directory = os.path.join(self.create_tempdir().full_path, "ckpt")
    ocp.save(directory, {"emb": jnp.arange(40, dtype=jnp.int32).reshape(10, 4)})

    plan = surgery.pipeline(surgery.resize(r"^emb$", axis=0, size=4))
    target = {"emb": _sds((4, 4), jnp.int32)}
    result = surgery.load(directory, plan, target=target)

    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["emb"])),
        np.arange(40).reshape(10, 4)[:4],
    )

  def test_per_shard_reads_disjoint_regions(self):
    directory = os.path.join(self.create_tempdir().full_path, "ckpt")
    ocp.save(directory, {"w": jnp.arange(12, dtype=jnp.float32).reshape(4, 3)})

    devices = jax.devices()
    if len(devices) >= 2:
      mesh, spec = Mesh(np.asarray(devices[:2]), ("x",)), PartitionSpec("x")
    else:
      mesh, spec = Mesh(np.asarray(devices), ("x",)), PartitionSpec()
    sharding = NamedSharding(mesh, spec)

    plan = surgery.pipeline()
    target = {"w": _sds((4, 3), jnp.float32, sharding=sharding)}
    result = surgery.load(directory, plan, target=target)

    self.assertEqual(result["w"].sharding, sharding)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["w"])),
        np.arange(12, dtype=np.float32).reshape(4, 3),
    )


class ShardedExecutionTest(absltest.TestCase):

  def _mesh(self):
    devices = jax.devices()
    if len(devices) >= 2:
      return Mesh(np.asarray(devices[:2]), ("x",)), PartitionSpec("x")
    return Mesh(np.asarray(devices), ("x",)), PartitionSpec()

  def test_stack_assembles_per_shard(self):
    source = {
        "e.0.w": jnp.array([0, 1, 2], dtype=jnp.float32),
        "e.1.w": jnp.array([3, 4, 5], dtype=jnp.float32),
        "e.2.w": jnp.array([6, 7, 8], dtype=jnp.float32),
        "e.3.w": jnp.array([9, 10, 11], dtype=jnp.float32),
    }
    mesh, spec = self._mesh()
    sharding = NamedSharding(mesh, spec)
    plan = surgery.pipeline(surgery.stack(r"e\.(\d+\.)", axis=0))
    target = {"e.w": _sds((4, 3), jnp.float32, sharding=sharding)}

    result = surgery.load(source, plan, target=target)

    out = result["e.w"]
    self.assertEqual(out.sharding, sharding)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(out)),
        np.arange(12, dtype=np.float32).reshape(4, 3),
    )


if __name__ == "__main__":
  absltest.main()
