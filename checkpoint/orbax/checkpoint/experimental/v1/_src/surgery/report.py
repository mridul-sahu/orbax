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

"""The dry-run account produced by resolving a plan."""

import dataclasses

# Error kinds. A plan collects every defect it finds rather than raising on the
# first, so a single preview call returns the complete repair list.
MISSING_SOURCE = "missing_source"
INCOMPLETE_GROUP = "incomplete_group"
SHAPE_MISMATCH = "shape_mismatch"
DTYPE_MISMATCH = "dtype_mismatch"
UNMATCHED_RULE = "unmatched_rule"
UNCOVERED_TARGET = "uncovered_target"
OVERLAPPING_ASSIGNMENTS = "overlapping_assignments"
ORPHANED_OPTIMIZER_STATE = "orphaned_optimizer_state"
UNEXPECTED_KEY = "unexpected_key"
RENAME_COLLISION = "rename_collision"


@dataclasses.dataclass(frozen=True)
class PlanError:
  """A single defect found while resolving a plan.

  Attributes:
    kind: One of the module-level kind constants.
    key: The manifest or target key the defect concerns.
    detail: Human-readable text naming the pattern or group at fault.
  """

  kind: str
  key: str
  detail: str


class SurgeryError(Exception):
  """Raised when a plan is resolved with unresolved errors present."""


@dataclasses.dataclass
class PlanReport:
  """A complete account of what a plan will do, before any array data moves.

  Attributes:
    moved: Ordered (old key, new key) pairs produced by renames and imports.
    stacked: Target key to index-to-source-key mapping for each stack.
    fused: Target key to the ordered source keys that were concatenated.
    dropped: Keys the plan removes; these are never read.
    filled: Target key to the reason it is initialized rather than loaded.
    errors: Every defect found; empty on a healthy plan.
  """

  moved: list[tuple[str, str]] = dataclasses.field(default_factory=list)
  stacked: dict[str, dict[int, str]] = dataclasses.field(default_factory=dict)
  fused: dict[str, list[str]] = dataclasses.field(default_factory=dict)
  dropped: list[str] = dataclasses.field(default_factory=list)
  filled: dict[str, str] = dataclasses.field(default_factory=dict)
  errors: list[PlanError] = dataclasses.field(default_factory=list)

  def add_error(self, kind: str, key: str, detail: str) -> None:
    """Records a single defect.

    Args:
      kind: One of the module-level kind constants.
      key: The manifest or target key the defect concerns.
      detail: Human-readable text naming the pattern or group at fault.
    """
    self.errors.append(PlanError(kind=kind, key=key, detail=detail))

  def raise_if_errors(self) -> None:
    """Raises `SurgeryError` listing every collected defect, if any."""
    if not self.errors:
      return
    lines = [f"  [{e.kind}] {e.key}: {e.detail}" for e in self.errors]
    raise SurgeryError(
        f"Plan resolution found {len(self.errors)} error(s):\n"
        + "\n".join(lines)
    )

  def __str__(self) -> str:
    parts = [
        f"moved={len(self.moved)}",
        f"stacked={len(self.stacked)}",
        f"fused={len(self.fused)}",
        f"dropped={len(self.dropped)}",
        f"filled={len(self.filled)}",
        f"errors={len(self.errors)}",
    ]
    out = "PlanReport(" + ", ".join(parts) + ")"
    for e in self.errors:
      out += f"\n  [{e.kind}] {e.key}: {e.detail}"
    return out
