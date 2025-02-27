# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from .dino_clstoken_loss import DINOLoss
from .ibot_patch_loss import iBOTPatchLoss
from .koleo_loss import KoLeoLoss
from .hierarchical_ce import HierarchicalCrossEntropyLoss, get_weighting, load_hierarchy, load_hierarchy_v2