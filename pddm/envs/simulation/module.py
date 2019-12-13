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

"""Module for caching Python modules related to simulation."""

import sys

_MUJOCO_PY_MODULE = None

_GLFW_MODULE = None


def get_mujoco_py():
    """Returns the mujoco_py module."""
    global _MUJOCO_PY_MODULE
    if _MUJOCO_PY_MODULE:
        return _MUJOCO_PY_MODULE
    try:
        import mujoco_py
        # Override the warning function.
        from mujoco_py.builder import cymj
        cymj.set_warning_callback(_mj_warning_fn)
    except ImportError:
        print(
            'Failed to import mujoco_py. Ensure that mujoco_py (using MuJoCo '
            'v1.50) is installed.',
            file=sys.stderr)
        sys.exit(1)
    _MUJOCO_PY_MODULE = mujoco_py
    return mujoco_py


def get_mujoco_py_mjlib():
    """Returns the mujoco_py mjlib module."""

    class MjlibDelegate:
        """Wrapper that forwards mjlib calls."""

        def __init__(self, lib):
            self._lib = lib

        def __getattr__(self, name: str):
            if name.startswith('mj'):
                return getattr(self._lib, '_' + name)
            raise AttributeError(name)

    return MjlibDelegate(get_mujoco_py().cymj)


def _mj_warning_fn(warn_data: bytes):
    """Warning function override for mujoco_py."""
    print('WARNING: Mujoco simulation is unstable (has NaNs): {}'.format(
        warn_data.decode()))
