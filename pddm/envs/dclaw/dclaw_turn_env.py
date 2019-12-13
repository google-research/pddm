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

import os
import numpy as np
from gym import utils
from mujoco_py import load_model_from_path, MjSim

from pddm.envs import mujoco_env
from pddm.envs.robot import Robot
from pddm.envs.utils.quatmath import euler2quat
from pddm.utils.helper_funcs import angle_difference


class DClawTurnEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, n_jnt=9, n_obj=1, frame_skip=40):

        #init vars
        self.target_bid = 0
        self.object_qp = 0 # Object joint angle
        self.object_qp_init = 0
        self.object_qv_init = 0
        self.goal = [0] # target joint angle
        self.obs_dict = {}
        self.rewards_dict = {}
        self.time = 0

        assert n_jnt > 0
        assert n_obj >= 0
        self.n_jnt = n_jnt
        self.n_obj = n_obj
        self.n_dofs = n_jnt + n_obj

        xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'dclaw_valve3.xml')

        # Make robot
        self.robot=Robot(
                n_jnt=n_jnt,
                n_obj=n_obj,
                n_dofs = self.n_dofs,
                pos_bounds=[[-np.pi / 2.0, np.pi / 2.0]] * 10,
                vel_bounds=[[-5, 5]] * 10,
                )

        # Initialize mujoco env
        self.initializing = True
        super().__init__(
            xml_path,
            frame_skip=frame_skip,
            camera_settings=dict(
                distance=1.0,
                azimuth=140,
                elevation=-50,
            ),
        )
        utils.EzPickle.__init__(self)
        self.initializing = False
        self.skip = self.frame_skip

        # init target and action ranges
        self.target_bid = self.sim.model.body_name2id("target")
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:self.n_jnt,1]-self.model.actuator_ctrlrange[:self.n_jnt,0])

        # starting pose of dclaw
        self.init_qpos[[0, 3, 6]] = 0
        self.init_qpos[[1, 4, 7]] = -1.0
        self.init_qpos[[2, 5, 8]] = 1.35


    def get_reward(self, observations, actions):

        """get rewards of a given (observations, actions) pair

        Args:
            observations: (batchsize, obs_dim) or (obs_dim,)
            actions: (batchsize, ac_dim) or (ac_dim,)

        Return:
            r_total:
            done: True if env reaches terminal state (batchsize,1) or (1,)
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
        target_pos = observations[:,-1]
        screw_pos = observations[:,-3]
        joint_pos = observations[:, :self.n_jnt]
        joint_vel = observations[:, self.n_jnt:2*self.n_jnt]
        zeros = np.zeros(target_pos.shape)
        dones = zeros.copy() #this task is never terminated

        # screw position
        screw_dist = np.abs(angle_difference(screw_pos, target_pos))

        # reward for proximity to the target
        self.reward_dict['r_target_dist'] = -10* screw_dist
        bonus_small = zeros
        bonus_big = zeros
        bonus_small[screw_dist < 0.25] = 1
        bonus_big[screw_dist < 0.1] = 10
        self.reward_dict['bonus_small'] = bonus_small
        self.reward_dict['bonus_big'] = bonus_big

        # total reward
        self.reward_dict['r_total'] = self.reward_dict['r_target_dist'] + \
                                    self.reward_dict['bonus_small'] + \
                                    self.reward_dict['bonus_big']

        #return
        if not batch_mode:
            return self.reward_dict['r_total'][0], dones[0]
        return self.reward_dict['r_total'], dones

    def get_score(self, obs):
        target_pos = obs[-1]
        screw_pos = obs[-3]
        score = -1*np.abs(angle_difference(screw_pos, target_pos))

        return score

    def step(self, a):

        a = np.clip(a, -1.0, 1.0) # Clamp the actions to limits

        if not self.initializing:
            a = self.act_mid + a*self.act_rng # mean center and scale

        # apply actions and step
        self.robot.step(self, a[:self.n_jnt], step_duration=self.frame_skip*self.model.opt.timestep)

        # obs/rew/done/score
        obs = self._get_obs()
        reward, done = self.get_reward(obs, a)
        score = self.get_score(obs)

        # finalize step
        env_info = {'time': self.time,
                    'obs_dict': self.obs_dict,
                    'rewards': self.reward_dict,
                    'score': score}

        return obs, reward, done, env_info

    def _get_obs(self):

        # get obs
        self.robot.get_obs(self, robot_noise_ratio=0, object_noise_ratio=0)
        time, hand_qp, hand_qv, lid_qp, lid_qv = self.robot.get_obs_from_cache(self, -1)
        self.time = time

        # update target pose
        desired_orien = np.zeros(3)
        desired_orien[2] = self.goal[0]

        # update target visualization
        self.model.body_quat[self.target_bid] = euler2quat(desired_orien)

        # populate obs dictionary
        self.obs_dict = {}
        self.obs_dict['hand_qp'] = hand_qp
        self.obs_dict['hand_qv'] = hand_qv
        self.obs_dict['lid_qp'] = lid_qp
        self.obs_dict['lid_qv'] = lid_qv
        self.obs_dict['goal'] = self.goal

        return np.concatenate([ self.obs_dict['hand_qp'],
                                self.obs_dict['hand_qv'],
                                self.obs_dict['lid_qp'],
                                self.obs_dict ['lid_qv'],
                                self.obs_dict['goal'] ])

    def reset_model(self):

        # valve target angle
        rand_angle = self.np_random.uniform(low=np.pi/2, high=np.pi)
        rand_sign = np.random.randint(2)
        if rand_sign==0:
            self.goal = [rand_angle]
        else:
            self.goal = [-rand_angle]

        # valve start angle
        self.object_qp = self.np_random.uniform(low=-np.pi/4, high=np.pi/4)

        #reset the joints and screw to set location/velocity
        self.reset_pose = np.append(self.init_qpos[:self.n_jnt], self.object_qp)
        self.reset_vel = np.append(self.init_qvel[:self.n_jnt], self.object_qv_init)
        self.reset_goal = self.goal.copy()

        return self.do_reset(self.reset_pose, self.reset_vel, self.reset_goal)

    def do_reset(self, reset_pose, reset_vel, reset_goal):

        #reset hand and object
        self.robot.reset(self, reset_pose, reset_vel)
        self.sim.forward()

        #reset target
        self.goal = reset_goal.copy()

        #return
        return self._get_obs()