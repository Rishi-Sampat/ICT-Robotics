# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *


import os

# This gets the directory where this __init__.py file lives
EXTENSION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Optional: Define a specific path for assets to make configs cleaner
ICT_BOT_ASSETS_DIR = os.path.join(EXTENSION_DIR, "assets")