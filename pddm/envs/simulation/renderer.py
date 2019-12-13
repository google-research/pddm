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

import abc
import enum
import sys
from typing import Dict, Optional

import numpy as np

from pddm.envs.simulation import module

# Default window dimensions.
DEFAULT_WINDOW_WIDTH = 1024
DEFAULT_WINDOW_HEIGHT = 768

DEFAULT_WINDOW_TITLE = 'MuJoCo Viewer'

_MAX_RENDERBUFFER_SIZE = 2048


class RenderMode(enum.Enum):
    """Rendering modes for offscreen rendering."""
    RGB = 0
    DEPTH = 1
    SEGMENTATION = 2


class Renderer(abc.ABC):
    """Base interface for rendering simulations."""

    def __init__(self, camera_settings: Optional[Dict] = None):
        self._camera_settings = camera_settings

    @abc.abstractmethod
    def close(self):
        """Cleans up any resources being used by the renderer."""

    @abc.abstractmethod
    def render_to_window(self):
        """Renders the simulation to a window."""

    @abc.abstractmethod
    def render_offscreen(self,
                         width: int,
                         height: int,
                         mode: RenderMode = RenderMode.RGB,
                         camera_id: int = -1) -> np.ndarray:
        """Renders the camera view as a NumPy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default, uses
                the free camera.

        Returns:
            A NumPy array of the pixels.
        """

    def _update_camera(self, camera):
        """Updates the given camera to move to the initial settings."""
        if not self._camera_settings:
            return
        distance = self._camera_settings.get('distance')
        azimuth = self._camera_settings.get('azimuth')
        elevation = self._camera_settings.get('elevation')
        lookat = self._camera_settings.get('lookat')

        if distance is not None:
            camera.distance = distance
        if azimuth is not None:
            camera.azimuth = azimuth
        if elevation is not None:
            camera.elevation = elevation
        if lookat is not None:
            camera.lookat[:] = lookat


class MjPyRenderer(Renderer):
    """Class for rendering mujoco_py simulations."""

    def __init__(self, sim, **kwargs):
        assert isinstance(sim, module.get_mujoco_py().MjSim), \
            'MjPyRenderer takes a mujoco_py MjSim object.'
        super().__init__(**kwargs)
        self._sim = sim
        self._onscreen_renderer = None
        self._offscreen_renderer = None

    def render_to_window(self):
        """Renders the simulation to a window."""
        if not self._onscreen_renderer:
            self._onscreen_renderer = module.get_mujoco_py().MjViewer(self._sim)
            self._update_camera(self._onscreen_renderer.cam)

        self._onscreen_renderer.render()

    def render_offscreen(self,
                         width: int,
                         height: int,
                         mode: RenderMode = RenderMode.RGB,
                         camera_id: int = -1) -> np.ndarray:
        """Renders the camera view as a NumPy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default, uses
                the free camera.

        Returns:
            A NumPy array of the pixels.
        """
        if not self._offscreen_renderer:
            self._offscreen_renderer = module.get_mujoco_py() \
                .MjRenderContextOffscreen(self._sim)

        # Update the camera configuration for the free-camera.
        if camera_id == -1:
            self._update_camera(self._offscreen_renderer.cam)

        self._offscreen_renderer.render(width, height, camera_id)
        if mode == RenderMode.RGB:
            data = self._offscreen_renderer.read_pixels(
                width, height, depth=False)
            # Original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == RenderMode.DEPTH:
            data = self._offscreen_renderer.read_pixels(
                width, height, depth=True)[1]
            # Original image is upside-down, so flip it
            return data[::-1, :]
        else:
            raise NotImplementedError(mode)

    def close(self):
        """Cleans up any resources being used by the renderer."""