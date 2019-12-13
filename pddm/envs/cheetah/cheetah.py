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

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):

        self.time = 0

        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

        self.skip = self.frame_skip

    def get_reward(self, observations, actions):

        """get rewards of a given (observations, actions) pair

        Args:
            observations: (batchsize, obs_dim) or (obs_dim,)
            actions: (batchsize, ac_dim) or (ac_dim,)

        Return:
            r_total: (batchsize,1) or (1,), reward for that pair
            done: (batchsize,1) or (1,), True if reaches terminal state
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
        xvel = observations[:, 9]
        body_angle = observations[:, 2]

        #calc rew
        self.reward_dict['actions'] = -0.1 * np.sum(np.square(actions), axis=1)
        self.reward_dict['run'] = xvel
        self.reward_dict['r_total'] = self.reward_dict['actions'] + self.reward_dict['run']

        #check if done
        dones = np.zeros((observations.shape[0],))
        dones[body_angle>1.0] = 1

        #return
        if not batch_mode:
            return self.reward_dict['r_total'][0], dones[0]
        return self.reward_dict['r_total'], dones


    def get_score(self, obs):
        xposafter = obs[0]
        return xposafter


    def step(self, action):

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

        self.obs_dict = {}
        self.obs_dict['joints_pos'] = self.sim.data.qpos.flat.copy()
        self.obs_dict['joints_vel'] = self.sim.data.qvel.flat.copy()

        return np.concatenate([
            self.obs_dict['joints_pos'], #9
            self.obs_dict['joints_vel'], #9
        ])


    def reset_model(self):

        # set reset pose/vel
        self.reset_pose = self.init_qpos + self.np_random.uniform(
                        low=-.1, high=.1, size=self.model.nq)
        self.reset_vel = self.init_qvel + self.np_random.randn(self.model.nv) * .1

        #reset the env to that pose/vel
        return self.do_reset(self.reset_pose.copy(), self.reset_vel.copy())


    def do_reset(self, reset_pose, reset_vel, reset_goal=None):

        #reset
        self.set_state(reset_pose, reset_vel)

        #return
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.distance = 12