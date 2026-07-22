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

"""Public API for plan-based model surgery.

Surgery describes a checkpoint transformation as an ordered pipeline of
declarative operations, resolves it against metadata before any data is read,
and executes it as a sharded read that lands directly in the target buffers::

  from orbax.checkpoint.experimental.v1 import surgery

  pipe = surgery.pipeline(
      surgery.rename((r"^model\\.", "")),
      surgery.fuse(parts=("mlp.gate.w", "mlp.up.w"), into="mlp.gate_up.w"),
      surgery.stack(r"layers\\.(\\d+\\.)", axis=0),
      surgery.cast(r".*", jnp.bfloat16),
  )
  report = surgery.preview(pipe, source=ocp.metadata(path), target=abstract)
  state = surgery.load(path, pipe, target=abstract)
"""

# pylint: disable=g-importing-member, unused-import, g-multiple-import

from orbax.checkpoint.experimental.v1._src.surgery.execution import execute
from orbax.checkpoint.experimental.v1._src.surgery.execution import load
from orbax.checkpoint.experimental.v1._src.surgery.execution import read_keys
from orbax.checkpoint.experimental.v1._src.surgery.execution import save
from orbax.checkpoint.experimental.v1._src.surgery.operations import cast
from orbax.checkpoint.experimental.v1._src.surgery.operations import compute
from orbax.checkpoint.experimental.v1._src.surgery.operations import drop
from orbax.checkpoint.experimental.v1._src.surgery.operations import fuse
from orbax.checkpoint.experimental.v1._src.surgery.operations import mirror
from orbax.checkpoint.experimental.v1._src.surgery.operations import rename
from orbax.checkpoint.experimental.v1._src.surgery.operations import repeat
from orbax.checkpoint.experimental.v1._src.surgery.operations import resize
from orbax.checkpoint.experimental.v1._src.surgery.operations import select
from orbax.checkpoint.experimental.v1._src.surgery.operations import split
from orbax.checkpoint.experimental.v1._src.surgery.operations import stack
from orbax.checkpoint.experimental.v1._src.surgery.operations import take
from orbax.checkpoint.experimental.v1._src.surgery.operations import unstack
from orbax.checkpoint.experimental.v1._src.surgery.pipeline import pipeline
from orbax.checkpoint.experimental.v1._src.surgery.pipeline import Plan
from orbax.checkpoint.experimental.v1._src.surgery.pipeline import preview
from orbax.checkpoint.experimental.v1._src.surgery.pipeline import ResolvedPlan
from orbax.checkpoint.experimental.v1._src.surgery.report import PlanError
from orbax.checkpoint.experimental.v1._src.surgery.report import PlanReport
from orbax.checkpoint.experimental.v1._src.surgery.report import SurgeryError
