# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for loading MuJoCo models."""

import os
from typing import Dict, Optional

from pddm.envs.simulation import module
from pddm.envs.simulation.renderer import MjPyRenderer, RenderMode


class MujocoSimRobot:
    """Class that encapsulates a MuJoCo simulation.

    This class exposes methods that are agnostic to the simulation backend.
    Two backends are supported:
    1. mujoco_py - MuJoCo v1.50
    2. dm_control - MuJoCo v2.00
    """

    def __init__(self,
                 model_file: str,
                 camera_settings: Optional[Dict] = None):
        """Initializes a new simulation.

        Args:
            model_file: The MuJoCo XML model file to load.
            camera_settings: Settings to initialize the renderer's camera. This
              can contain the keys `distance`, `azimuth`, and `elevation`.
        """

        if not os.path.isfile(model_file):
            raise ValueError(
                '[MujocoSimRobot] Invalid model file path: {}'.format(
                    model_file))

        mujoco_py = module.get_mujoco_py()
        self.model = mujoco_py.load_model_from_path(model_file)
        self.sim = mujoco_py.MjSim(self.model)
        self.renderer = MjPyRenderer(
            self.sim, camera_settings=camera_settings)

        self.data = self.sim.data

    def close(self):
        """Cleans up any resources being used by the simulation."""
        self.renderer.close()

    def save_binary(self, path: str):
        """Saves the loaded model to a binary .mjb file."""
        if os.path.exists(path):
            raise ValueError(
                '[MujocoSimRobot] Path already exists: {}'.format(path))
        if not path.endswith('.mjb'):
            path = path + '.mjb'

        with open(path, 'wb') as f:
            f.write(self.model.get_mjb())

    def get_mjlib(self):
        """Returns an object that exposes the low-level MuJoCo API."""
        return module.get_mujoco_py_mjlib()

    def _patch_mjlib_accessors(self, model, data):
        """Adds accessors to the DM Control objects to support mujoco_py API."""
