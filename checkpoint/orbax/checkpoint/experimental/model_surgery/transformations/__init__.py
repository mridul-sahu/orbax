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

"""Model surgery transformations."""

# pylint: disable=g-importing-member, g-multiple-import, unused-import

from orbax.checkpoint.experimental.model_surgery.transformations.fusing import fuse_by_keys
from orbax.checkpoint.experimental.model_surgery.transformations.fusing import fuse_by_pattern
from orbax.checkpoint.experimental.model_surgery.transformations.nesting import unflatten
from orbax.checkpoint.experimental.model_surgery.transformations.renaming import rename_by_regex
from orbax.checkpoint.experimental.model_surgery.transformations.repeating import repeat_by_keys
from orbax.checkpoint.experimental.model_surgery.transformations.repeating import repeat_by_pattern
from orbax.checkpoint.experimental.model_surgery.transformations.stacking import stack
from orbax.checkpoint.experimental.model_surgery.transformations.types import Transformation
