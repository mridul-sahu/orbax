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

"""The dry-run account produced by resolving a surgery function."""

import dataclasses

# Error kinds. Resolution collects every defect it finds rather than raising on
# the first, so a single preview returns the complete repair list.
MISSING_KEY = "missing_key"
UNEXPECTED_KEY = "unexpected_key"
SHAPE_MISMATCH = "shape_mismatch"
DTYPE_MISMATCH = "dtype_mismatch"


@dataclasses.dataclass(frozen=True)
class PlanError:
  """A single defect found while resolving a plan.

  Attributes:
    kind: One of the module-level kind constants.
    key: The target key the defect concerns.
    detail: Human-readable description.
  """

  kind: str
  key: str
  detail: str


class SurgeryError(Exception):
  """Raised when a plan with unresolved errors is executed."""


@dataclasses.dataclass
class PlanReport:
  """What a plan will do, before any array data moves.

  Attributes:
    materialized: Target key to the primitives that force a full read of their
      inputs before any shard of that key can be placed.
    errors: Every defect found; empty on a healthy plan.
  """

  materialized: dict[str, list[str]] = dataclasses.field(default_factory=dict)
  errors: list[PlanError] = dataclasses.field(default_factory=list)

  def add_error(self, kind: str, key: str, detail: str) -> None:
    self.errors.append(PlanError(kind=kind, key=key, detail=detail))

  def raise_if_errors(self) -> None:
    """Raises `SurgeryError` listing every collected defect, if any."""
    if self.errors:
      raise SurgeryError(str(self))

  def __str__(self) -> str:
    out = (
        f"PlanReport(materialized={len(self.materialized)},"
        f" errors={len(self.errors)})"
    )
    for e in self.errors:
      out += f"\n  [{e.kind}] {e.key}: {e.detail}"
    return out
