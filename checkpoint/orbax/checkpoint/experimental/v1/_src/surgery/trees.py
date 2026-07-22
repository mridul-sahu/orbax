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

"""Flat-key views over source and target trees.

Surgery addresses leaves by a single dotted key. These helpers move between a
nested tree and that flat view while preserving the original structure, so a
merge can draw from nested source metadata and hand back a tree shaped exactly
like the abstract target.
"""

from typing import Any

import jax

SEP = "."


def _entry_str(entry: Any) -> str:
  if isinstance(entry, jax.tree_util.DictKey):
    return str(entry.key)
  if isinstance(entry, jax.tree_util.SequenceKey):
    return str(entry.idx)
  if isinstance(entry, jax.tree_util.GetAttrKey):
    return str(entry.name)
  if isinstance(entry, jax.tree_util.FlattenedIndexKey):
    return str(entry.key)
  return str(entry)


def _key_str(path) -> str:
  return SEP.join(_entry_str(e) for e in path)


def flatten(tree: Any) -> dict[str, Any]:
  """Returns a flat dotted-key view of `tree`.

  Args:
    tree: A nested tree of leaves (arrays, shape-dtype structs, metadata).

  Returns:
    A mapping from dotted key to leaf.
  """
  paths_and_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
  return {_key_str(path): leaf for path, leaf in paths_and_leaves}


def keys_in_order(tree: Any) -> list[str]:
  """Returns the dotted keys of `tree` in its canonical leaf order."""
  paths_and_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
  return [_key_str(path) for path, _ in paths_and_leaves]


def unflatten_like(tree: Any, flat: dict[str, Any]) -> Any:
  """Rebuilds a tree shaped like `tree` from a flat dotted-key mapping.

  Args:
    tree: The tree whose structure the result must match.
    flat: A mapping from dotted key to the leaf value to place.

  Returns:
    A tree with the structure of `tree` and leaves taken from `flat`.
  """
  paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
  leaves = [flat[_key_str(path)] for path, _ in paths_and_leaves]
  return jax.tree_util.tree_unflatten(treedef, leaves)
