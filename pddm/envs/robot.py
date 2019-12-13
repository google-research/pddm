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

import numpy as np
from collections import deque
from collections import namedtuple

# obervations structure
observation = namedtuple(
    'observation',
    ['time', 'qpos_robot', 'qvel_robot', 'qpos_object', 'qvel_object'])


class Robot(object):

    def __init__(self, n_jnt, n_obj, n_dofs, pos_bounds=None, vel_bounds=None, **kwargs):

        self.n_jnt = n_jnt
        self.n_obj = n_obj
        self.n_dofs = n_dofs

        self.has_obj = False
        if self.n_obj>0:
            self.has_obj = True

        # Cache that gets updated
        self.observation_cache_maxsize = 5
        self.observation_cache = deque([], maxlen=self.observation_cache_maxsize)

        # Pos and vel bounds
        self.pos_bounds = None
        if pos_bounds is not None:
            pos_bounds = np.array(pos_bounds, dtype=np.float32)
            assert pos_bounds.shape == (self.n_dofs, 2)
            for low, high in pos_bounds:
                assert low < high
            self.pos_bounds = pos_bounds
        self.vel_bounds = None
        if vel_bounds is not None:
            vel_bounds = np.array(vel_bounds, dtype=np.float32)
            assert vel_bounds.shape == (self.n_dofs, 2)
            for low, high in vel_bounds:
                assert low < high
            self.vel_bounds = vel_bounds


    # refresh the observation cache
    def _observation_cache_refresh(self, env):
        for _ in range(self.observation_cache_maxsize):
            self.get_obs(env, robot_noise_ratio=0, object_noise_ratio=0)

    # get past observation
    def get_obs_from_cache(self, env, index=-1):
        assert (index>=0 and index<self.observation_cache_maxsize) or \
                (index<0 and index>=-self.observation_cache_maxsize), \
                "cache index out of bound. (cache size is %2d)"%self.observation_cache_maxsize
        obs = self.observation_cache[index]
        if self.has_obj:
            return obs.time, obs.qpos_robot, obs.qvel_robot, obs.qpos_object, obs.qvel_object
        else:
            return obs.time, obs.qpos_robot, obs.qvel_robot

    # get observation
    def get_obs(self, env, robot_noise_ratio=0.05, object_noise_ratio=0.05):
        qp = env.sim.data.qpos[:self.n_jnt].ravel()
        qv = env.sim.data.qvel[:self.n_jnt].ravel()
        if self.has_obj:
            qp_obj = env.sim.data.qpos[-self.n_obj:].ravel()
            qv_obj = env.sim.data.qvel[-self.n_obj:].ravel()
        else:
            qp_obj = None
            qv_obj = None
        self.time = env.sim.data.time

        # Simulate observation noise
        if not env.initializing:
            noise_amp = robot_noise_ratio*(env.model.jnt_range[:self.n_jnt,1]-env.model.jnt_range[:self.n_jnt,0])
            qp += noise_amp*env.np_random.uniform(low=-.5, high=.5, size=self.n_jnt)
            qv += noise_amp*env.np_random.uniform(low=-.5, high=.5, size=self.n_jnt)
            if self.has_obj:
                noise_amp = object_noise_ratio*(env.model.jnt_range[-self.n_obj:,1]-env.model.jnt_range[-self.n_obj:,0])
                qp_obj += noise_amp*env.np_random.uniform(low=-.5, high=.5, size=self.n_obj)
                qv_obj += noise_amp*env.np_random.uniform(low=-.5, high=.5, size=self.n_obj)

        # cache observations
        obs = observation(
            time=self.time,
            qpos_robot=qp,
            qvel_robot=qv,
            qpos_object=qp_obj,
            qvel_object=qv_obj)
        self.observation_cache.append(obs)

        if self.has_obj:
            return obs.time, obs.qpos_robot, obs.qvel_robot, obs.qpos_object, obs.qvel_object
        else:
            return obs.time, obs.qpos_robot, obs.qvel_robot

    # clip only joint position limits
    # since we can only control those anyway
    def ctrl_position_limits(self, ctrl_position):
        ctrl_feasible_position = np.clip(ctrl_position,
                                         self.pos_bounds[:self.n_jnt, 0],
                                         self.pos_bounds[:self.n_jnt, 1])
        return ctrl_feasible_position

    # enforce velocity limits.
    def enforce_velocity_limits(self, ctrl_position, step_duration):
        last_obs = self.observation_cache[-1]
        desired_vel = (ctrl_position[:self.n_jnt] - last_obs.qpos_robot[:self.n_jnt])/step_duration

        feasible_vel = np.clip(desired_vel, self.vel_bounds[:self.n_jnt, 0], self.vel_bounds[:self.n_jnt, 1])
        feasible_position = last_obs.qpos_robot + feasible_vel*step_duration
        return feasible_position

    # step the robot env
    def step(self, env, ctrl_desired, step_duration):

        # Populate observation cache during startup
        if env.initializing:
            self._observation_cache_refresh(env)

        # enforce velocity limits
        ctrl_feasible = self.enforce_velocity_limits(ctrl_desired, step_duration)

        # enforce position limits
        ctrl_feasible = self.ctrl_position_limits(ctrl_feasible)

        # Send controls to the robot
        env.do_simulation(ctrl_feasible, int(step_duration/env.sim.model.opt.timestep))  # render is folded in here

        return 1

    # clip the whole thing
    def clip_positions(self, positions):
        assert len(positions) == self.n_jnt or len(positions) == self.n_dofs
        pos_bounds = self.pos_bounds[:len(positions)]
        return np.clip(positions, pos_bounds[:, 0], pos_bounds[:, 1])

    def reset(self, env, reset_pose, reset_vel):

        reset_pose = self.clip_positions(reset_pose)

        # env.sim.reset()
        env.sim.data.qpos[:self.n_jnt] = reset_pose[:self.n_jnt].copy()
        env.sim.data.qvel[:self.n_jnt] = reset_vel[:self.n_jnt].copy()
        if self.has_obj:
            env.sim.data.qpos[-self.n_obj:] = reset_pose[-self.n_obj:].copy()
            env.sim.data.qvel[-self.n_obj:] = reset_vel[-self.n_obj:].copy()
        env.sim.forward()

        # refresh observation cache before exit
        self._observation_cache_refresh(env)