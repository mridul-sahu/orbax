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

"""Model surgery written as an ordinary JAX function.

A surgery function takes the source tree, as the checkpoint stores it, and
returns the target tree::

  from orbax.checkpoint.experimental.v1 import surgery

  def convert(p):
    experts = [p[f"layers.0.experts.{i}.w"].T for i in range(8)]
    return {"moe.w": jnp.stack(experts).astype(jnp.bfloat16)}

  report = surgery.preview(convert, ocp.metadata(path).metadata, abstract)
  state = surgery.load(path, convert, target=abstract)

The function is traced on shapes alone. Each device shard then reads only the
source bytes that land in it, and the result equals `convert` applied to the
loaded checkpoint, bit for bit.

Two things can be customized per call. `rules` decides how each JAX primitive
becomes boxes: `DEFAULT_RULES.with_layout(name, rule)` adds or replaces a rule,
built from the functions in `surgery.rules`, and `.without(name)` makes a
primitive compute whole. `backend` decides where box values are computed:
`HostBackend` lays them out in host memory, `DeviceBackend` on the target
device with XLA; by default CPU targets use the host and accelerators the
device. Both backends accept per-primitive overrides through `apply=`.
"""

# pylint: disable=g-importing-member, unused-import, g-multiple-import

from orbax.checkpoint.experimental.v1._src.surgery import rules
from orbax.checkpoint.experimental.v1._src.surgery.device import DeviceBackend
from orbax.checkpoint.experimental.v1._src.surgery.execution import load
from orbax.checkpoint.experimental.v1._src.surgery.fill import Backend
from orbax.checkpoint.experimental.v1._src.surgery.host import HostBackend
from orbax.checkpoint.experimental.v1._src.surgery.reads import LoadStats
from orbax.checkpoint.experimental.v1._src.surgery.report import PlanError
from orbax.checkpoint.experimental.v1._src.surgery.report import PlanReport
from orbax.checkpoint.experimental.v1._src.surgery.report import SurgeryError
from orbax.checkpoint.experimental.v1._src.surgery.rules import DEFAULT_RULES
from orbax.checkpoint.experimental.v1._src.surgery.rules import Rules
from orbax.checkpoint.experimental.v1._src.surgery.tracing import preview
from orbax.checkpoint.experimental.v1._src.surgery.tracing import read_keys
