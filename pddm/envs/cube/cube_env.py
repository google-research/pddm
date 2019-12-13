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
from gym import utils
import os

from pddm.envs import mujoco_env
from pddm.envs.robot import Robot
from pddm.envs.utils.quatmath import euler2quat, quatDiff2Vel

class CubeEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, n_jnt=24, frame_skip=40,):

        # init vars
        self.time = 0
        self.counter = 0
        self.target_x = 0
        self.target_y = 0
        self.target_z = 0
        self.grasp_sid = 0
        self.obs_dict = {}
        self.reward_dict = {}
        self.frame_skip = frame_skip
        self.skip = frame_skip

        # dims
        assert n_jnt > 0
        self.n_jnt = n_jnt
        self.n_obj_dofs = 9 ## 6 cube, 9 for cube + target joints
        self.n_dofs = self.n_jnt + self.n_obj_dofs

        xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'shadowhand_hand_cube.xml')

        # Make robot
        self.robot=Robot(
                n_jnt=self.n_jnt,
                n_obj=self.n_obj_dofs,
                n_dofs = self.n_dofs,
                pos_bounds=[[-40, 40]] * self.n_dofs, #dummy
                vel_bounds=[[-3, 3]] * self.n_dofs,
                )

        # Initialize mujoco env
        self.startup = True
        self.initializing = True
        super().__init__(
            xml_path,
            frame_skip=frame_skip,
            camera_settings=dict(
                distance=0.9,
                azimuth=-40,
                elevation=-50,
            ),
        )
        utils.EzPickle.__init__(self)
        self.startup = False
        self.initializing = False

        #init joint and site ids
        self.target_x = self.sim.model.joint_name2id('targetRx')
        self.target_y = self.sim.model.joint_name2id('targetRy')
        self.target_z = self.sim.model.joint_name2id('targetRz')
        self.grasp_sid = self.sim.model.site_name2id('S_finger_grasp')

        # reset position
        self.init_qpos = self.sim.model.key_qpos[0].copy()

        # action ranges
        self.act_mid = self.init_qpos[:self.n_jnt]
        self.upper_rng = (self.model.actuator_ctrlrange[:,1]-self.act_mid)
        self.lower_rng = (self.act_mid-self.model.actuator_ctrlrange[:,0])


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

        # obs:
        # self.obs_dict['robot_pos'], #24
        # self.obs_dict['object_position'], #3
        # self.obs_dict['object_orientation'], #3
        # self.obs_dict['object_velp'], #3
        # self.obs_dict['object_velr'], #3
        # self.obs_dict['desired_orientation'], #3

        #get vars
        obj_pos = observations[:, (24):(24)+3]
        obj_orientation = observations[:,(24+3):(24+3)+3]
        desired_orientation = observations[:,-3:]
        obj_height = observations[:,24+2]
        zeros = np.zeros(obj_height.shape)

        #orientation
        angle_diffs = np.linalg.norm(obj_orientation - desired_orientation, axis=1)

        #fall
        is_fall = zeros.copy()
        is_fall[obj_height < -0.1] = 1

        #done based on is_fall
        dones = (is_fall==1) if not self.startup else zeros

        #rewards
        self.reward_dict['ori_dist'] = -7*angle_diffs
        self.reward_dict['drop_penalty'] = -1000*is_fall
        self.reward_dict['r_total'] = self.reward_dict['ori_dist'] + self.reward_dict['drop_penalty']

        #return
        if not batch_mode:
            return self.reward_dict['r_total'][0], dones[0]
        return self.reward_dict['r_total'], dones


    def get_score(self, obs_dict):
        return -1.0*np.linalg.norm(obs_dict['object_orientation'] - obs_dict['desired_orientation'])

    def step(self, a):

        if self.startup:
            self.desired_pose = np.array([0,0,0])
        else:
            self.desired_pose = self.goal[self.counter].copy()

        #############
        ### ACTIONS
        ##############

        # clip and scale action
        a = np.clip(a, -1.0, 1.0)
        if self.startup:
            # only for the initialization phase
            a = a
        else:
            # mean center and scale
            # a = self.act_mid + a*self.act_rng
            a[a>0] = self.act_mid[a>0] + a[a>0]*self.upper_rng[a>0]
            a[a<=0] = self.act_mid[a<=0] + a[a<=0]*self.lower_rng[a<=0]

        # take the action
        self.robot.step(self, a, step_duration=self.skip*self.model.opt.timestep)

        ##################
        ### OBS AND REWARD
        ##################

        # get obs/rew/done/score
        obs = self._get_obs()
        reward, done = self.get_reward(obs, a)
        score = self.get_score(self.obs_dict)

        #return
        env_info = {'time': self.time,
                    'obs_dict': self.obs_dict,
                    'rewards':self.reward_dict,
                    'score':score}
        self.counter +=1
        return obs, reward, done, env_info


    def _get_obs(self):

        #update target pose visualization
        self.data.qpos[self.target_x] = self.desired_pose[0]
        self.data.qpos[self.target_y] = self.desired_pose[1]
        self.data.qpos[self.target_z] = self.desired_pose[2]

        #get obs
        self.robot.get_obs(self, robot_noise_ratio=0, object_noise_ratio=0)
        t, qp_hand, qv_hand, qp_obj, qv_obj = self.robot.get_obs_from_cache(self, -1)

        self.time = t
        self.obs_dict = {}
        self.obs_dict['robot_pos']          = qp_hand.copy() #24

        self.obs_dict['object_position']    = qp_obj[:3].copy() #3 (x/y/z) translation pos of cube
        self.obs_dict['object_orientation'] = qp_obj[3:6].copy() #3 (r/p/y) rotational pos of cube

        self.obs_dict['object_velp']        = qv_obj[:3].copy() #3 translational vel of cube
        self.obs_dict['object_velr']        = qv_obj[3:6].copy() #3 rotational vel of cube

        self.obs_dict['desired_orientation'] = self.desired_pose.copy() #3 desired (r/p/y) orientation of cube

        return np.concatenate([ self.obs_dict['robot_pos'], #24
                            self.obs_dict['object_position'], #3
                            self.obs_dict['object_orientation'], #3
                            self.obs_dict['object_velp'], #3
                            self.obs_dict['object_velr'], #3
                            self.obs_dict['desired_orientation'], #3
                            ])

    def reset_model(self):
        self.reset_pose = self.init_qpos.copy()
        self.reset_vel = self.init_qvel.copy()
        self.reset_goal = self.create_goal_trajectory()
        return self.do_reset(self.reset_pose, self.reset_vel, self.reset_goal)

    def do_reset(self, reset_pose, reset_vel, reset_goal=None):

        # reset counts
        self.counter=0

        #### reset goal
        if reset_goal is None:
            self.goal = self.create_goal_trajectory()
        else:
            self.goal = reset_goal.copy()

        # reset hand and objects
        self.robot.reset(self, reset_pose, reset_vel)
        self.sim.forward()
        return self._get_obs()

    def create_goal_trajectory(self):

        len_of_goals = 1000

        #####################################
        #####################################

        #rotate 20 deg about y axis (cos(a/2), sin(a/2), 0, 0) (up/down)
        #rotate 20 deg about z axis (cos(a/2), 0, 0, sin(a/2)) (left/right)
        left = [0, 0, -1.5]
        right = [0, 0, 1.5]
        up = [1.5, 0, 0]
        down = [-1.5, 0, 0]

        half_up = [0.7, 0, 0]
        half_down = [-0.7, 0, 0]
        half_left = [0, 0, -0.7]
        half_right = [0, 0, 0.7]

        slight_up = [0.35, 0, 0]
        slight_down = [-0.35, 0, 0]
        slight_left = [0, 0, -0.35]
        slight_right = [0, 0, 0.35]

        #####################################
        #####################################

        # CHOOSE one of these goal options here:
        # goal_options = [half_up, half_down, half_left, half_right, slight_right, slight_left, slight_down, slight_up, left, right, up, down]
        # goal_options = [half_up, half_down, half_left, half_right]
        # goal_options = [half_up, half_down, half_left, half_right, slight_right, slight_left, slight_down, slight_up]
        # goal_options = [left, right]
        # goal_options = [up, down]
        goal_options = [left, right, up, down]

        #####################################
        #####################################

        # A single rollout consists of alternating between 2 (same or diff) goals:
        same_goals = True
        if same_goals:
            goal_selected1 = np.random.randint(len(goal_options))
            goal_selected2 = goal_selected1
        else:
            goal_selected1= np.random.randint(len(goal_options))
            goal_selected2= np.random.randint(len(goal_options))
        goals = [goal_options[goal_selected1], goal_options[goal_selected2]]

        # Create list of these goals
        time_per_goal = 35
        step_num = 0
        curr_goal_num = 0
        goal_traj = []
        while step_num<len_of_goals:
            goal_traj.append(np.tile(goals[curr_goal_num], (time_per_goal, 1)))
            if curr_goal_num==0:
                curr_goal_num=1
            else:
                curr_goal_num=0
            step_num+=time_per_goal

        goal_traj = np.concatenate(goal_traj)
        return goal_traj