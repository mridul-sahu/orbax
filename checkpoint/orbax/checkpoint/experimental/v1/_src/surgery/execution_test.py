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

"""End-to-end loads, in memory and from checkpoints."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.v1 import surgery
import orbax.checkpoint.experimental.v1 as ocp
import safetensors.numpy

Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
PartitionSpec = jax.sharding.PartitionSpec


def _sds(shape, dtype, sharding=None):
  return jax.ShapeDtypeStruct(tuple(shape), np.dtype(dtype), sharding=sharding)


def _host(x):
  return np.asarray(jax.device_get(x))


def _leading_sharding():
  devices = np.asarray(jax.devices()[:2])
  spec = PartitionSpec("x") if len(devices) == 2 else PartitionSpec()
  return NamedSharding(Mesh(devices, ("x",)), spec)


class InMemoryLoadTest(absltest.TestCase):

  def test_rename_fuse_stack_cast(self):
    source = {
        "model.layers.0.mlp.gate.w": jnp.array([1, 2], dtype=jnp.float32),
        "model.layers.0.mlp.up.w": jnp.array([3, 4], dtype=jnp.float32),
        "model.layers.1.mlp.gate.w": jnp.array([5, 6], dtype=jnp.float32),
        "model.layers.1.mlp.up.w": jnp.array([7, 8], dtype=jnp.float32),
    }

    def fn(p):
      fused = [
          jnp.concatenate(
              [
                  p[f"model.layers.{i}.mlp.gate.w"],
                  p[f"model.layers.{i}.mlp.up.w"],
              ]
          )
          for i in range(2)
      ]
      return {"layers.mlp.gate_up.w": jnp.stack(fused).astype(jnp.float16)}

    target = {"layers.mlp.gate_up.w": _sds((2, 4), jnp.float16)}
    out = _host(surgery.load(source, fn, target=target)["layers.mlp.gate_up.w"])
    self.assertEqual(out.dtype, np.dtype(jnp.float16))
    np.testing.assert_array_equal(
        out, np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float16)
    )

  def test_flat_result_nests_to_the_target_structure(self):
    source = {
        "encoder.w": jnp.arange(4, dtype=jnp.float32),
        "decoder.w": jnp.arange(4, dtype=jnp.float32) + 4,
    }
    target = {
        "encoder": {"w": _sds((4,), jnp.float32)},
        "decoder": {"w": _sds((4,), jnp.float32)},
    }
    result = surgery.load(source, lambda p: p, target=target)
    np.testing.assert_array_equal(
        _host(result["decoder"]["w"]), np.arange(4, dtype=np.float32) + 4
    )

  def test_stack_assembles_per_shard(self):
    source = {
        f"e.{i}.w": jnp.arange(3, dtype=jnp.float32) + 3 * i for i in range(4)
    }
    sharding = _leading_sharding()
    fn = lambda p: {"e.w": jnp.stack([p[f"e.{i}.w"] for i in range(4)])}
    target = {"e.w": _sds((4, 3), jnp.float32, sharding=sharding)}

    out = surgery.load(source, fn, target=target)["e.w"]

    self.assertEqual(out.sharding, sharding)
    np.testing.assert_array_equal(
        _host(out), np.arange(12, dtype=np.float32).reshape(4, 3)
    )

  def test_missing_and_unexpected_keys_are_errors(self):
    source = {"a": jnp.zeros(2), "b": jnp.zeros(2)}
    report = surgery.preview(
        lambda p: p,
        source,
        {"a": _sds((2,), jnp.float32), "c": _sds((2,), jnp.float32)},
    )
    self.assertCountEqual(
        [(e.kind, e.key) for e in report.errors],
        [("missing_key", "c"), ("unexpected_key", "b")],
    )
    with self.assertRaises(surgery.SurgeryError):
      report.raise_if_errors()


class CheckpointLoadTest(absltest.TestCase):

  def _save(self, name, tree):
    directory = os.path.join(self.create_tempdir().full_path, name)
    ocp.save(directory, tree)
    return directory

  def test_dropped_key_is_not_read(self):
    directory = self._save(
        "ckpt",
        {
            "kept": jnp.arange(4, dtype=jnp.int32),
            "dropped": jnp.arange(4, dtype=jnp.int32) + 100,
        },
    )
    fn = lambda p: {"kept": p["kept"]}

    self.assertEqual(
        surgery.read_keys(fn, ocp.metadata(directory).metadata), {"kept"}
    )
    result = surgery.load(directory, fn, target={"kept": _sds((4,), jnp.int32)})
    np.testing.assert_array_equal(_host(result["kept"]), np.arange(4))

  def test_truncation_reads_only_the_kept_rows(self):
    directory = self._save(
        "ckpt", {"emb": jnp.arange(40, dtype=jnp.int32).reshape(10, 4)}
    )
    result = surgery.load(
        directory,
        lambda p: {"emb": p["emb"][:4]},
        target={"emb": _sds((4, 4), jnp.int32)},
    )
    np.testing.assert_array_equal(
        _host(result["emb"]), np.arange(40).reshape(10, 4)[:4]
    )

  def test_per_shard_reads_disjoint_regions(self):
    directory = self._save(
        "ckpt", {"w": jnp.arange(12, dtype=jnp.float32).reshape(4, 3)}
    )
    sharding = _leading_sharding()
    target = {"w": _sds((4, 3), jnp.float32, sharding=sharding)}
    result = surgery.load(directory, lambda p: p, target=target)
    self.assertEqual(result["w"].sharding, sharding)
    np.testing.assert_array_equal(
        _host(result["w"]), np.arange(12, dtype=np.float32).reshape(4, 3)
    )

  def test_merge_two_checkpoints_with_a_fresh_leaf(self):
    enc = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    vision = self._save("vision", {"params.enc": enc, "params.head": enc})
    text = self._save("text", {"params.enc": enc + 10})

    def fn(p):
      return {
          "image_encoder": p["vision"]["params.enc"],
          "text_encoder": p["text"]["params.enc"],
          "fusion": jnp.zeros(4, jnp.float32),
      }

    target = {
        "image_encoder": _sds((2, 3), jnp.float32),
        "text_encoder": _sds((2, 3), jnp.float32),
        "fusion": _sds((4,), jnp.float32),
    }
    sources = {"vision": vision, "text": text}
    metadata = {k: ocp.metadata(v).metadata for k, v in sources.items()}
    self.assertEqual(
        surgery.read_keys(fn, metadata),
        {"vision.params.enc", "text.params.enc"},
    )
    result = surgery.load(sources, fn, target=target)

    np.testing.assert_array_equal(_host(result["image_encoder"]), enc)
    np.testing.assert_array_equal(_host(result["text_encoder"]), enc + 10)
    np.testing.assert_array_equal(_host(result["fusion"]), np.zeros(4))


class SafetensorsLoadTest(absltest.TestCase):

  def _layout(self, layout):
    return ocp.Context(checkpoint_layout=layout)

  def test_sharded_strided_transpose_reads_one_file(self):
    path = os.path.join(self.create_tempdir().full_path, "model.safetensors")
    q = np.arange(96, dtype=np.float32).reshape(12, 8)
    safetensors.numpy.save_file(
        {"model.q": q, "unused": np.zeros(5, np.float32)}, path
    )
    fn = lambda p: {"q": p["model.q"][::3].T}
    target = {"q": _sds((8, 4), jnp.float32, sharding=_leading_sharding())}

    with self._layout(ocp.options.CheckpointLayout.SAFETENSORS):
      self.assertEqual(
          surgery.read_keys(fn, ocp.metadata(path).metadata), {"model.q"}
      )
      result = surgery.load(path, fn, target=target)

    np.testing.assert_array_equal(_host(result["q"]), q[::3].T)

  def test_directory_of_files_with_column_shards(self):
    directory = self.create_tempdir().full_path
    gate = np.arange(64, dtype=np.float16).reshape(8, 8)
    up = -np.arange(64, dtype=np.float16).reshape(8, 8)
    safetensors.numpy.save_file(
        {"gate": gate}, os.path.join(directory, "a.safetensors")
    )
    safetensors.numpy.save_file(
        {"up": up}, os.path.join(directory, "b.safetensors")
    )
    devices = np.asarray(jax.devices()[:2])
    columns = NamedSharding(
        Mesh(devices, ("x",)),
        PartitionSpec(None, "x") if len(devices) == 2 else PartitionSpec(),
    )
    fn = lambda p: {"wi": jnp.concatenate([p["gate"], p["up"]], axis=1)}
    target = {"wi": _sds((8, 16), jnp.float16, sharding=columns)}

    with self._layout(ocp.options.CheckpointLayout.SAFETENSORS):
      result = surgery.load(directory, fn, target=target)

    np.testing.assert_array_equal(
        _host(result["wi"]), np.concatenate([gate, up], axis=1)
    )

  def test_merge_safetensors_with_orbax_checkpoint(self):
    root = self.create_tempdir().full_path
    hf = os.path.join(root, "hf.safetensors")
    safetensors.numpy.save_file(
        {"embed": np.arange(6, dtype=np.float32).reshape(3, 2)}, hf
    )
    orbax = os.path.join(root, "orbax")
    ocp.save(orbax, {"head": jnp.ones((2, 3), jnp.float32)})

    def fn(p):
      return {"embed": p["hf"]["embed"], "head": p["orbax"]["head"].T}

    target = {
        "embed": _sds((3, 2), jnp.float32),
        "head": _sds((3, 2), jnp.float32),
    }
    with self._layout(ocp.options.CheckpointLayout.AUTO):
      result = surgery.load({"hf": hf, "orbax": orbax}, fn, target=target)

    np.testing.assert_array_equal(
        _host(result["embed"]), np.arange(6).reshape(3, 2)
    )
    np.testing.assert_array_equal(_host(result["head"]), np.ones((3, 2)))


if __name__ == "__main__":
  absltest.main()
