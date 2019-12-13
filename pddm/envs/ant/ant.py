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
import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env

from pddm.envs.utils.quatmath import quat_to_euler


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):


    def __init__(self,
                 xml_file='ant.xml',
                 ctrl_cost_weight=0.5,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 reset_noise_scale=0.1,
                 contact_force_range=0,
                 exclude_current_positions_from_observation=True):

        utils.EzPickle.__init__(**locals())

        self.time = 0

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self.min_z, self.max_z = self._healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        self.startup = True
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        self.startup = False

        self.skip = self.frame_skip

        #self.action_space.high 1, self.action_space.low -1
        #actions are control limited right now (not force limited)
        #self.model.actuator_gear starts at [150,0,0,0,0,0] for each of the 8 actuators
        for i in range(len(self.model.actuator_gear)):
            self.model.actuator_gear[i][0]/=5

    def get_reward(self, observations, actions):

        """get rewards of a given (observations, actions) pair

        Args:
            observations: (batchsize, obs_dim) or (obs_dim,)
            actions: (batchsize, ac_dim) or (ac_dim,)

        Return:
            r_total: reward for that pair: (batchsize,1) or (1,)
            done: True if env reaches terminal state: (batchsize,1) or (1,)
        """

        #initialize and reshape as needed, for batch mode
        self.reward_dict = {}
        if len(observations.shape)==1:
            observations = np.expand_dims(observations, axis = 0)
            actions = np.expand_dims(actions, axis = 0)
            batch_mode = False
        else:
            batch_mode = True

        #get vars
        xvel = observations[:, -1]
        height = observations[:, -2]
        roll_angle = observations[:, 0]
        pitch_angle = observations[:, 1]

        #is flipped
        is_flipping = np.zeros((observations.shape[0],))
        is_flipping[np.abs(roll_angle) > 0.7] = 1
        is_flipping[np.abs(pitch_angle) > 0.6] = 1

        #check health
        all_finite = np.isfinite(observations).all(axis=1)
        is_healthy = np.ones((observations.shape[0],))
        is_healthy[all_finite==False] = 0
        is_healthy[height < self.min_z] = 0
        is_healthy[height > self.max_z] = 0
        is_healthy[is_flipping==True] = 0

        #calc rew
        self.reward_dict['actions'] = -self._ctrl_cost_weight * np.sum(np.square(actions), axis=1)
        self.reward_dict['run'] = 10*xvel
        self.reward_dict['health'] = is_healthy*self._healthy_reward
        self.reward_dict['flipping'] = -500*is_flipping
        self.reward_dict['r_total'] = self.reward_dict['run'] + self.reward_dict['health'] + self.reward_dict['flipping'] ### + self.reward_dict['actions']

        #check if done
        dones = np.zeros((observations.shape[0],))
        if self._terminate_when_unhealthy:
            dones[is_healthy==False] = 1

        #return
        if not batch_mode:
            return self.reward_dict['r_total'][0], dones[0]
        return self.reward_dict['r_total'], dones

    def get_score(self, obs):
        xvel = obs[-1]
        return xvel

    def step(self, action):

        self.prev_com_pos = self.get_body_com("torso").copy()

        #step
        self.do_simulation(action, self.frame_skip)

        #obs/reward/done/score
        ob = self._get_obs()
        rew, done = self.get_reward(ob, action)
        score = self.get_score(ob)

        #return
        env_info = {'time': self.time,
                    'obs_dict': self.obs_dict,
                    'rewards': self.reward_dict,
                    'score': score}
        return ob, rew, done, env_info

    def _get_obs(self):

        #com vel
        if(self.startup):
            xvel = [0.0]
        else:
            curr_com_pos = self.get_body_com("torso").copy()
            prev_com_pos = self.prev_com_pos
            xvel = [(curr_com_pos-prev_com_pos)[0]/self.dt]

        #data.qpos is 15
            # 3 com pos
            # 4 com quat
            # 8 : 4 pairs of hip/ankle (start top R, go ccw)

        self.obs_dict = {}
        self.obs_dict['com_angular_pose'] = quat_to_euler(self.sim.data.qpos[3:7]) # 3
        self.obs_dict['com_pos'] = self.sim.data.qpos[:3] # 3
        self.obs_dict['joints_pos'] = self.sim.data.qpos[7:].copy() # 15 --> 8
        self.obs_dict['joints_vel'] = self.sim.data.qvel[-8:].copy() # 14 --> 8
        self.obs_dict['com_vel_x'] = xvel.copy() #1

        if self._exclude_current_positions_from_observation:
            return np.concatenate([
                self.obs_dict['com_angular_pose'],
                self.obs_dict['joints_pos'],
                self.obs_dict['joints_vel'],
                [self.obs_dict['com_pos'][2]], #only height
                self.obs_dict['com_vel_x'],
            ])
        else:
            return np.concatenate([
                self.obs_dict['com_angular_pose'],
                self.obs_dict['joints_pos'],
                self.obs_dict['joints_vel'],
                self.obs_dict['com_pos'], # x y and z
                self.obs_dict['com_vel_x'],
            ])

    def reset_model(self):

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # set reset pose/vel
        self.reset_pose = self.init_qpos + self.np_random.uniform(
                        low=noise_low, high=noise_high, size=self.model.nq)
        self.reset_pose[3:7] = np.array([1,0,0,0]) # this is a quaternion
        self.reset_vel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(self.model.nv)

        #reset the env to that pose/vel
        return self.do_reset(self.reset_pose.copy(), self.reset_vel.copy())


    def do_reset(self, reset_pose, reset_vel, reset_goal=None):

        #reset
        self.set_state(reset_pose, reset_vel)

        #return
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = 15
