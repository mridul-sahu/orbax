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


"""Checkpoint sources: resolving a layout and reading tensor sub-ranges.

A checkpoint is opened through the layout the Orbax context selects, exactly as
`ocp.load` resolves it. Each layout has an opener that turns a tensor name into
a sub-range reader: Orbax tensors are read as TensorStore slices, SafeTensors
tensors as byte ranges of their row-major block. `OPENERS` maps a layout class
to its opener, so another layout can be supported by adding an entry.
"""

from collections.abc import Callable, Sequence
import functools
import json
import os
from typing import Any

import numpy as np
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.path import format_utils
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_v0_layout
from orbax.checkpoint.experimental.v1._src.layout import registry as layout_registry
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
from orbax.checkpoint.experimental.v1._src.surgery import boxes
from orbax.checkpoint.experimental.v1._src.surgery import rules as rules_lib
from orbax.checkpoint.experimental.v1._src.surgery import tracing
import tensorstore as ts

Region = boxes.Region

# Opens one tensor of a checkpoint, by name, as a sub-range reader.
Opener = Callable[[str], boxes.ReadFn]

# Two byte runs of one SafeTensors read are fetched together when the gap
# between them is at most this or the length of the later run, so a read
# fetches at most twice the bytes it needs and never splits across a page.
_MIN_SAFETENSORS_GAP = 4096


def is_path(source: Any) -> bool:
  return isinstance(source, (str, os.PathLike))


def is_path_mapping(source: Any) -> bool:
  return (
      isinstance(source, dict)
      and bool(source)
      and all(is_path(v) for v in source.values())
  )


def _orbax_opener(path: Any, name: str | None, metadata: Any) -> Opener:
  """Opens tensors of an Orbax checkpoint as TensorStore slices."""
  del metadata
  directory = path / name if name else path
  use_ocdbt = format_utils.is_ocdbt_checkpoint(directory)
  recorded = json.loads(
      (directory / format_utils.PYTREE_METADATA_FILE).read_text()
  )
  use_zarr3 = recorded.get("use_zarr3", False)
  context = ts_utils.get_ts_context(use_ocdbt=use_ocdbt)

  def open_tensor(key: str) -> boxes.ReadFn:
    spec = ts_utils.ArrayReadSpec(
        directory.as_posix(), key, use_zarr3, use_ocdbt=use_ocdbt
    ).json
    store = ts.open(ts.Spec(spec), open=True, context=context).result()
    return lambda region: np.asarray(store[region.slices].read().result())

  return open_tensor


def _coalesce(
    runs: Sequence[tuple[int, int]],
) -> list[tuple[int, int, list[tuple[int, int]]]]:
  """Groups offset-sorted byte runs into `(start, stop, runs)` file reads."""
  blocks: list[tuple[int, int, list[tuple[int, int]]]] = []
  for offset, length in runs:
    if blocks and offset - blocks[-1][1] <= max(_MIN_SAFETENSORS_GAP, length):
      start, _, members = blocks[-1]
      members.append((offset, length))
      blocks[-1] = (start, offset + length, members)
    else:
      blocks.append((offset, offset + length, [(offset, length)]))
  return blocks


def _read_safetensors_region(
    file: Any,
    base: int,
    shape: tuple[int, ...],
    dtype: np.dtype,
    region: Region,
) -> np.ndarray:
  """Reads one region of a tensor stored row-major at byte `base` of `file`.

  Args:
    file: The `.safetensors` file.
    base: The absolute byte offset of the tensor's first element.
    shape: The tensor's shape.
    dtype: The tensor's dtype.
    region: The region to read; a strided region reads its dense bounds and
      keeps every step-th element.

  Returns:
    The region's values.
  """
  bounds = tuple(zip(region.start, region.stop))
  runs = safetensors_layout.index_domain_to_byte_runs(
      bounds, shape, dtype.itemsize, base
  )
  out = bytearray(sum(length for _, length in runs))
  pos = 0
  with file.open("rb") as f:
    for start, stop, members in _coalesce(runs):
      f.seek(start)
      block = memoryview(f.read(stop - start))
      for offset, length in members:
        out[pos : pos + length] = block[
            offset - start : offset - start + length
        ]
        pos += length
  dense = np.frombuffer(out, dtype=dtype).reshape(
      [hi - lo for lo, hi in bounds]
  )
  return dense[tuple(slice(None, None, s) for s in region.steps)]


def _safetensors_opener(
    path: Any, name: str | None, metadata: dict[str, Any]
) -> Opener:
  """Opens tensors of a SafeTensors checkpoint as byte-range readers."""
  del name
  files = asyncio_utils.run_sync(safetensors_layout.build_file_index(path))

  def open_tensor(key: str) -> boxes.ReadFn:
    entry = files[key]
    return functools.partial(
        _read_safetensors_region,
        entry.path,
        entry.data_start + entry.info["data_offsets"][0],
        tuple(metadata[key].shape),
        np.dtype(metadata[key].dtype),
    )

  return open_tensor


# Maps a layout class to its opener: (path, pytree name, flat metadata) ->
# tensor name -> sub-range reader.
OPENERS: dict[type[Any], Callable[[Any, str | None, Any], Opener]] = {
    orbax_layout.OrbaxLayout: _orbax_opener,
    orbax_v0_layout.OrbaxV0Layout: _orbax_opener,
    safetensors_layout.SafetensorsLayout: _safetensors_opener,
}


def _resolve_checkpoint(path: Any) -> tuple[dict[str, Any], Opener]:
  """Resolves a checkpoint's layout as `ocp.load` does.

  Args:
    path: The checkpoint path.

  Returns:
    The checkpoint's flat metadata, and how to open each of its tensors.
  """
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)
  resolver = asyncio_utils.run_sync(
      layout_registry.CheckpointLayoutResolver.resolve(
          path,
          ctx.checkpoint_layout,
          pytree_name=checkpoint_layout.AUTO_CHECKPOINTABLE_KEY,
      )
  )
  layout, name = resolver.layout, resolver.pytree_name
  metadata = tree_utils.to_flat_dict(
      asyncio_utils.run_sync(layout.metadata(path, name)).metadata, sep="."
  )
  opener = OPENERS.get(type(layout))
  if opener is None:
    raise ValueError(
        f"Surgery cannot read {type(layout).__name__} checkpoints; known"
        f" layouts are {[t.__name__ for t in OPENERS]}."
    )
  return metadata, opener(path, name, metadata)


def open_checkpoints(
    source: Any,
    fn: Callable[[Any], Any],
    rules: rules_lib.Rules = rules_lib.DEFAULT_RULES,
) -> Any:
  """Builds the source tree for a checkpoint path or a mapping of them.

  A mapping hands `fn` one flat tree per name, so a function can merge several
  checkpoints, of either layout.

  Args:
    source: A checkpoint path, or a mapping from name to checkpoint path.
    fn: The surgery function, traced on metadata to find the keys it reads.
    rules: How each primitive becomes boxes.

  Returns:
    The flat source tree, or a mapping from name to flat source tree. Each
    key `fn` reads holds a `StoredArray`; every other key keeps its metadata.
  """
  paths = {"": source} if is_path(source) else source
  resolved = {name: _resolve_checkpoint(path) for name, path in paths.items()}
  metadata = {name: flat for name, (flat, _) in resolved.items()}
  needed = tracing.read_keys(
      fn, metadata[""] if is_path(source) else metadata, rules=rules
  )
  opened = {}
  for name, (flat, open_tensor) in resolved.items():
    prefix = f"{name}." if name else ""
    opened[name] = {
        key: (
            boxes.StoredArray(
                shape=tuple(meta.shape),
                dtype=np.dtype(meta.dtype),
                read=open_tensor(key),
            )
            if prefix + key in needed
            else meta
        )
        for key, meta in flat.items()
    }
  return opened[""] if is_path(source) else opened
