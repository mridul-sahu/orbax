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

"""Loads must reproduce the surgery function exactly, reading less."""

import math
import os
import random
import re

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.v1 import surgery
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.surgery import boxes
from orbax.checkpoint.experimental.v1._src.surgery import tracing

Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
PartitionSpec = jax.sharding.PartitionSpec


def _random(shape, dtype, seed=0):
  rng = np.random.default_rng(seed)
  return jnp.asarray(rng.standard_normal(shape).astype(np.float32)).astype(
      dtype
  )


def _eager(fn, source):
  with jax.disable_jit():
    return fn(source)


def _target_like(tree, sharding_for=None):
  def spec(x):
    sharding = sharding_for(x) if sharding_for else None
    return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding)

  return jax.tree.map(spec, tree)


def _mesh():
  return Mesh(np.asarray(jax.devices()), ("x",))


def _shard_leading(x):
  n = len(jax.devices())
  spec = PartitionSpec("x") if x.shape and x.shape[0] % n == 0 else None
  return NamedSharding(_mesh(), spec or PartitionSpec())


class _RecordingStore:
  """Source arrays that record every region a load reads."""

  def __init__(self, arrays):
    self.arrays = {k: np.asarray(v) for k, v in arrays.items()}
    self.reads: dict[str, list[boxes.Region]] = {}

  def tree(self):
    def read_for(key):
      def read(region):
        self.reads.setdefault(key, []).append(region)
        return self.arrays[key][region.slices]

      return read

    return {
        k: boxes.StoredArray(shape=a.shape, dtype=a.dtype, read=read_for(k))
        for k, a in self.arrays.items()
    }

  def elements_read(self, key):
    return sum(r.size for r in self.reads.get(key, []))


class LoadMatchesFunctionTest(parameterized.TestCase):
  """Runs every load with the default backend; subclasses force another."""

  backend = None

  def load(self, *args, **kwargs):
    return surgery.load(*args, backend=self.backend, **kwargs)

  def assert_matches_eager(self, fn, source, sharding_for=_shard_leading):
    expected = _eager(fn, source)
    target = _target_like(expected, sharding_for)
    got = self.load(source, fn, target=target)
    for key in expected:
      want = np.asarray(expected[key])
      have = np.asarray(jax.device_get(got[key]))
      self.assertEqual(have.dtype, want.dtype, key)
      self.assertEqual(have.shape, want.shape, key)
      self.assertEqual(have.tobytes(), want.tobytes(), key)
    return got

  @parameterized.named_parameters(
      ("transpose", lambda p: {"o": p["w"].T}),
      ("strided_reverse", lambda p: {"o": p["w"][1:15:3, ::-1]}),
      ("split_axis", lambda p: {"o": p["w"].reshape(4, 4, 8)}),
      ("merge_axes", lambda p: {"o": p["w"].reshape(4, 4, 8).reshape(16, 8)}),
      (
          "noncontiguous_merge",
          lambda p: {
              "o": p["w"].reshape(2, 8, 8).swapaxes(0, 1).reshape(8, 16)
          },
      ),
      ("row", lambda p: {"o": p["w"][3]}),
      ("pad", lambda p: {"o": jnp.pad(p["w"], ((2, 1), (0, 3)))}),
      ("pad_value", lambda p: {"o": jnp.pad(p["w"], 1, constant_values=7.0)}),
      (
          "crop_pad",
          lambda p: {
              "o": jax.lax.pad(
                  p["w"], jnp.zeros((), p["w"].dtype), ((-2, 3, 0),) * 2
              )
          },
      ),
      ("repeat", lambda p: {"o": jnp.repeat(p["w"], 3, axis=1)}),
      ("tile", lambda p: {"o": jnp.tile(p["w"], (2, 1))}),
      ("expand", lambda p: {"o": jnp.expand_dims(p["w"], 1)}),
      ("cast", lambda p: {"o": p["w"].astype(jnp.float32)}),
      ("scale", lambda p: {"o": p["w"] * 2 + 1}),
      ("split", lambda p: dict(zip("ab", jnp.split(p["w"], [5], axis=0)))),
      ("unstack", lambda p: {str(i): r for i, r in enumerate(p["w"][:3])}),
  )
  def test_single_primitive(self, fn):
    self.assert_matches_eager(fn, {"w": _random((16, 8), jnp.bfloat16)})

  def test_hf_style_import(self):
    bf = jnp.bfloat16
    n_experts = 4
    source = {
        "q": _random((32, 32), bf, 1),
        "k": _random((8, 32), bf, 2),
        "v": _random((8, 32), bf, 3),
        "embed": _random((100, 32), bf, 4),
        **{
            f"experts.{i}.gate": _random((64, 32), bf, 10 + i)
            for i in range(n_experts)
        },
        **{
            f"experts.{i}.up": _random((64, 32), bf, 20 + i)
            for i in range(n_experts)
        },
    }

    def fn(p):
      gate = jnp.stack([p[f"experts.{i}.gate"].T for i in range(n_experts)])
      up = jnp.stack([p[f"experts.{i}.up"].T for i in range(n_experts)])
      return {
          "moe.wi": jnp.concatenate([gate, up], axis=-1),
          "attn.qkv": jnp.concatenate([p["q"], p["k"], p["v"]]).T,
          "attn.k": jnp.repeat(p["k"], 4, axis=0),
          "embed": jnp.pad(p["embed"], ((0, 28), (0, 0))),
          "rope_q": p["q"].reshape(4, 2, 4, 32).swapaxes(1, 2).reshape(32, 32),
      }

    report = surgery.preview(fn, source, _target_like(_eager(fn, source)))
    self.assertEqual(report.errors, [])
    self.assertEqual(report.materialized, {})
    self.assert_matches_eager(fn, source)

  def test_fp8_dequantize_is_evaluated_per_shard(self):
    source = {
        "w": _random((64, 32), jnp.float8_e4m3fn),
        "scale_inv": _random((2, 1), jnp.float32, 1),
    }

    def fn(p):
      w = p["w"].astype(jnp.float32).reshape(2, 32, 32)
      return {
          "w": (
              (w * p["scale_inv"][:, :, None])
              .reshape(64, 32)
              .astype(jnp.bfloat16)
          )
      }

    self.assertEqual(surgery.preview(fn, source).materialized, {})
    self.assert_matches_eager(fn, source)

  def test_unsupported_primitive_is_materialized_and_exact(self):
    source = {
        "a": _random((8, 8), jnp.float32),
        "b": _random((8, 8), jnp.float32),
    }

    def fn(p):
      return {"mm": (p["a"] @ p["b"]).T, "kept": p["a"][:4]}

    report = surgery.preview(fn, source)
    self.assertEqual(report.materialized, {"mm": ["dot_general"]})
    self.assert_matches_eager(fn, source)

  def test_shard_reads_each_expert_once_and_skips_unused(self):
    n = len(jax.devices())
    store = _RecordingStore(
        {
            **{f"e.{i}": _random((6, 4), jnp.float32, i) for i in range(2 * n)},
            "unused": _random((1000,), jnp.float32),
        }
    )

    def fn(p):
      return {"experts": jnp.stack([p[f"e.{i}"].T for i in range(2 * n)])}

    target = {
        "experts": jax.ShapeDtypeStruct(
            (2 * n, 4, 6),
            jnp.float32,
            sharding=NamedSharding(_mesh(), PartitionSpec("x")),
        )
    }
    got = self.load(store.tree(), fn, target=target)

    self.assertNotIn("unused", store.reads)
    for i in range(2 * n):
      self.assertEqual(store.elements_read(f"e.{i}"), 24)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(got["experts"])),
        np.stack([store.arrays[f"e.{i}"].T for i in range(2 * n)]),
    )

  def test_column_shard_reads_only_its_columns(self):
    n = len(jax.devices())
    store = _RecordingStore({"w": _random((8 * n, 16), jnp.float32)})

    def fn(p):
      return {"w": p["w"].T}

    target = {
        "w": jax.ShapeDtypeStruct(
            (16, 8 * n),
            jnp.float32,
            sharding=NamedSharding(_mesh(), PartitionSpec(None, "x")),
        )
    }
    self.load(store.tree(), fn, target=target)

    self.assertLen(store.reads["w"], n)
    self.assertEqual(store.elements_read("w"), 8 * n * 16)

  @parameterized.named_parameters(
      ("every_other_row", lambda w: w[::2], 8 * 8),
      ("reversed_every_third", lambda w: w[::-3, ::2], 6 * 4),
      ("strided_transpose", lambda w: w[1::4].T, 4 * 8),
  )
  def test_strided_view_reads_only_kept_elements(self, op, kept):
    store = _RecordingStore({"w": _random((16, 8), jnp.float32)})
    fn = lambda p: {"o": op(p["w"])}
    expected = _eager(fn, {"w": jnp.asarray(store.arrays["w"])})
    got = self.load(store.tree(), fn, target=_target_like(expected))
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(got["o"])), np.asarray(expected["o"])
    )
    self.assertEqual(store.elements_read("w"), kept)

  @parameterized.named_parameters(
      (
          "rope_reorder",
          lambda q: q.reshape(4, 2, 4, 32).swapaxes(1, 2).reshape(32, 32),
      ),
      ("repeat", lambda q: jnp.repeat(q, 4, axis=0)),
      ("concat_with_itself", lambda q: jnp.concatenate([q, q * 2])),
  )
  def test_nearby_boxes_share_one_read(self, op):
    store = _RecordingStore({"q": _random((32, 32), jnp.float32)})
    fn = lambda p: {"o": op(p["q"])}
    expected = _eager(fn, {"q": jnp.asarray(store.arrays["q"])})
    self.assertGreater(
        len(tracing.resolve(fn, store.tree()).manifest["o"].assignments), 1
    )
    got = self.load(store.tree(), fn, target=_target_like(expected))
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(got["o"])), np.asarray(expected["o"])
    )
    self.assertLen(store.reads["q"], 1)
    self.assertEqual(store.elements_read("q"), 32 * 32)

  def test_checkpoint_strided_read(self):
    directory = os.path.join(self.create_tempdir().full_path, "ckpt")
    w = np.arange(96, dtype=np.float32).reshape(12, 8)
    ocp.save(directory, {"w": jnp.asarray(w)})
    fn = lambda p: {"w": p["w"][::3, ::-2]}
    result = self.load(
        directory,
        fn,
        target={"w": jax.ShapeDtypeStruct((4, 4), jnp.float32)},
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["w"])), w[::3, ::-2]
    )

  def test_key_surgery_records_nothing_and_skips_dropped_keys(self):
    store = _RecordingStore(
        {
            "model.embed": _random((8, 4), jnp.float32),
            **{
                f"model.layers.{i}.q": _random((4, 4), jnp.float32, i)
                for i in range(11)
            },
            "optimizer.mu": _random((8, 4), jnp.float32),
        }
    )
    layer_q = re.compile(r"model\.layers\.(\d+)\.q")

    def fn(p):
      found = sorted(
          (int(m[1]), v) for k, v in p.items() if (m := layer_q.fullmatch(k))
      )
      return {
          "embed": p["model.embed"],
          "layers.q": jnp.stack([v for _, v in found]),
      }

    self.assertEqual(surgery.preview(fn, store.tree()).materialized, {})
    expected = _eager(fn, {k: jnp.asarray(v) for k, v in store.arrays.items()})
    got = self.load(store.tree(), fn, target=_target_like(expected))
    for key in expected:
      np.testing.assert_array_equal(
          np.asarray(jax.device_get(got[key])), np.asarray(expected[key])
      )
    self.assertNotIn("optimizer.mu", store.reads)

  @parameterized.named_parameters(
      ("reorder", "ij->ji", 1, {}),
      ("contraction", "ij,jk->ik", 2, {"o": ["dot_general"]}),
      ("elementwise", "ij,ij->ij", 2, {"o": ["dot_general"]}),
      ("diagonal", "ii->i", 1, {"o": ["reduce_sum"]}),
  )
  def test_einsum(self, spec, arity, materialized):
    source = {f"x{i}": _random((6, 6), jnp.float32, i) for i in range(arity)}
    fn = lambda p: {"o": jnp.einsum(spec, *(p[f"x{i}"] for i in range(arity)))}
    self.assertEqual(surgery.preview(fn, source).materialized, materialized)
    self.assert_matches_eager(fn, source)

  def test_replicated_target_is_filled_once_and_stats_add_up(self):
    store = _RecordingStore({"w": _random((8, 4), jnp.float32)})
    replicated = NamedSharding(_mesh(), PartitionSpec())
    target = {
        "w": jax.ShapeDtypeStruct((4, 8), jnp.float32, sharding=replicated)
    }
    stats = surgery.LoadStats()
    got = self.load(
        store.tree(),
        lambda p: {"w": p["w"].T},
        target=target,
        stats=stats,
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(got["w"])), store.arrays["w"].T
    )
    self.assertEqual(store.elements_read("w"), 32)
    self.assertEqual((stats.reads, stats.read_bytes, stats.shards), (1, 128, 1))
    self.assertGreaterEqual(stats.total_seconds, stats.transfer_seconds)

  def test_shape_mismatch_is_reported_before_reading(self):
    store = _RecordingStore({"w": _random((4, 6), jnp.float32)})
    report = surgery.preview(
        lambda p: {"w": p["w"].T},
        store.tree(),
        {"w": jax.ShapeDtypeStruct((4, 6), jnp.float32)},
    )
    self.assertNotEqual(report.errors, [])
    self.assertEqual(store.reads, {})

  def test_checkpoint_source_reads_only_referenced_keys(self):
    directory = os.path.join(self.create_tempdir().full_path, "ckpt")
    ocp.save(
        directory,
        {
            "wq": jnp.arange(24, dtype=jnp.float32).reshape(4, 6),
            "dropped": jnp.zeros((3,), dtype=jnp.float32),
        },
    )
    fn = lambda p: {"wq_t": p["wq"].T[::-1]}

    read = surgery.read_keys(fn, ocp.metadata(directory).metadata)
    self.assertEqual(read, {"wq"})

    result = self.load(
        directory,
        fn,
        target={"wq_t": jax.ShapeDtypeStruct((6, 4), jnp.float32)},
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(result["wq_t"])),
        np.arange(24, dtype=np.float32).reshape(4, 6).T[::-1],
    )


def _transpose_step(rng, shape):
  perm = list(range(len(shape)))
  rng.shuffle(perm)
  return lambda x: jnp.transpose(x, perm)


def _reshape_step(rng, shape):
  size, dims = math.prod(shape), []
  while size > 1 and len(dims) < 3:
    d = rng.choice([d for d in range(2, size + 1) if size % d == 0])
    dims.append(d)
    size //= d
  new = tuple(dims) + ((size,) if size > 1 else ())
  return lambda x: x.reshape(new or (1,))


def _slice_step(rng, shape):
  axis = rng.randrange(len(shape))
  index = [slice(None)] * len(shape)
  index[axis] = slice(rng.randrange(shape[axis]), None, rng.choice([1, 2, 3]))
  return lambda x: x[tuple(index)]


def _rev_step(rng, shape):
  axis = rng.randrange(len(shape))
  return lambda x: jnp.flip(x, axis)


def _pad_step(rng, shape):
  widths = [(0, 0)] * len(shape)
  widths[rng.randrange(len(shape))] = (rng.randrange(3), rng.randrange(3))
  return lambda x: jnp.pad(x, widths, constant_values=-1)


def _concat_step(rng, shape):
  axis = rng.randrange(len(shape))
  return lambda x: jnp.concatenate([x, x * 2], axis=axis)


def _repeat_step(rng, shape):
  axis, times = rng.randrange(len(shape)), rng.choice([2, 3])
  return lambda x: jnp.repeat(x, times, axis=axis)


_STEPS = (
    _transpose_step,
    _reshape_step,
    _slice_step,
    _rev_step,
    _pad_step,
    _concat_step,
    _repeat_step,
)


def _random_layout_fn(rng: random.Random, shape: tuple[int, ...]):
  """Builds a random chain of layout operations valid for `shape`."""
  ops = []
  for _ in range(rng.randint(2, 6)):
    op = rng.choice(_STEPS)(rng, shape)
    shape = jax.eval_shape(op, jax.ShapeDtypeStruct(shape, jnp.float32)).shape
    ops.append(op)

  def fn(p):
    x = p["x"]
    for op in ops:
      x = op(x)
    return {"y": x}

  return fn


class RandomLayoutChainTest(parameterized.TestCase):

  backend = None

  def load(self, *args, **kwargs):
    return surgery.load(*args, backend=self.backend, **kwargs)

  @parameterized.parameters(range(40))
  def test_random_chain_matches_eager(self, seed):
    rng = random.Random(seed)
    shape = tuple(rng.choice([2, 3, 4, 6]) for _ in range(rng.randint(1, 3)))
    source = {
        "x": jnp.arange(math.prod(shape), dtype=jnp.float32).reshape(shape)
    }
    fn = _random_layout_fn(rng, shape)
    expected = _eager(fn, source)
    got = self.load(source, fn, target=_target_like(expected))
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(got["y"])), np.asarray(expected["y"])
    )


class DeviceBackendMatchesFunctionTest(LoadMatchesFunctionTest):
  """The same suite with every shard built on its device by XLA."""

  backend = surgery.DeviceBackend()


class DeviceBackendRandomLayoutChainTest(RandomLayoutChainTest):

  backend = surgery.DeviceBackend()


class RulesTest(parameterized.TestCase):

  def test_without_computes_a_primitive_whole(self):
    source = {"w": _random((8, 4), jnp.float32)}
    fn = lambda p: {"o": p["w"].T}
    rules = surgery.DEFAULT_RULES.without("transpose")
    self.assertEqual(
        surgery.preview(fn, source, rules=rules).materialized,
        {"o": ["transpose"]},
    )
    got = surgery.load(
        source, fn, target=_target_like(_eager(fn, source)), rules=rules
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(got["o"])), np.asarray(source["w"]).T
    )

  def test_custom_rule_turns_a_derived_primitive_into_boxes(self):
    source = {"w": _random((6, 4), jnp.float32)}
    fn = lambda p: {"o": p["w"][None].sum(axis=0)}
    self.assertEqual(
        surgery.preview(fn, source).materialized, {"o": ["reduce_sum"]}
    )

    def sum_over_unit_axes(ops, params, shape, dtype):
      del shape, dtype
      if any(ops[0].shape[a] != 1 for a in params["axes"]):
        raise surgery.rules.Unsupported()
      return [surgery.rules.squeeze(ops[0], params["axes"])]

    rules = surgery.DEFAULT_RULES.with_layout("reduce_sum", sum_over_unit_axes)
    self.assertEqual(surgery.preview(fn, source, rules=rules).materialized, {})
    got = surgery.load(
        source, fn, target=_target_like(_eager(fn, source)), rules=rules
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(got["o"])), np.asarray(source["w"])
    )

  def test_backend_override_replaces_one_primitive(self):
    source = {"w": _random((4, 4), jnp.float32)}
    fn = lambda p: {"o": p["w"] * 2}
    calls = []

    def mul(x, y, **params):
      del params
      calls.append(1)
      return np.multiply(x, y)

    backend = surgery.HostBackend(apply={"mul": mul})
    got = surgery.load(
        source, fn, target=_target_like(_eager(fn, source)), backend=backend
    )
    self.assertNotEmpty(calls)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(got["o"])), np.asarray(source["w"]) * 2
    )


if __name__ == "__main__":
  absltest.main()
