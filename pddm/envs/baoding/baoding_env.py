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
import enum

from pddm.envs import mujoco_env
from pddm.envs.robot import Robot
from pddm.envs.utils.quatmath import euler2quat, quatDiff2Vel

## Define the task enum
class Task(enum.Enum):
    MOVE_TO_LOCATION = 0
    BAODING_CW = 1
    BAODING_CCW = 2

## Define task
# WHICH_TASK = Task.MOVE_TO_LOCATION
WHICH_TASK = Task.BAODING_CCW

class BaodingEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, n_jnt=24, frame_skip=40,):

        #init vars
        self.which_task = Task(WHICH_TASK)
        self.time = 0
        self.counter = 0
        self.grasp_sid = 0
        self.object1_bid = 0
        self.object2_bid = 0
        self.target1_sid = 0
        self.target2_sid = 0
        self.new_task_sid = 0
        self.obs_dict = {}
        self.reward_dict = {}
        self.frame_skip = frame_skip
        self.skip = frame_skip

        # dims
        assert n_jnt > 0
        self.n_jnt = n_jnt
        self.n_obj_dofs = 2*7 # two free joints, each w 7 dof (3 pos, 4 quat)
        self.n_dofs = self.n_jnt + self.n_obj_dofs

        # xml file paths, based on task
        if self.which_task==Task.MOVE_TO_LOCATION:
            xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'shadowhand_baoding_1visible.xml')
        else:
            xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'shadowhand_baoding_2.xml')

            # init desired trajectory, for baoding
            self.x_radius = 0.02
            self.y_radius = 0.02*1.5*1.2
            self.center_pos = [0.005,0.055]

            # balls start at these angles
            # 1= yellow = bottom right
            # 2= pink = top left
            self.ball_1_starting_angle = 3*np.pi/4.0
            self.ball_2_starting_angle = -np.pi/4.0

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
                distance=0.7,
                azimuth=-60,
                elevation=-50,
            ),
        )
        utils.EzPickle.__init__(self)
        self.startup = False
        self.initializing = False

        #init target and body sites
        self.grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.object1_bid = self.sim.model.body_name2id('ball1')
        self.object2_bid = self.sim.model.body_name2id('ball2')
        self.target1_sid = self.model.site_name2id('target1_site')
        self.target2_sid = self.model.site_name2id('target2_site')
        if self.which_task == Task.MOVE_TO_LOCATION:
            self.new_task_sid =  self.model.site_name2id('new_task_site')

        #reset position
        self.init_qpos = self.sim.model.key_qpos[0].copy()

        #action ranges
        self.act_mid = self.init_qpos[:self.n_jnt].copy()
        self.upper_rng = 0.9*(self.model.actuator_ctrlrange[:,1]-self.act_mid)
        self.lower_rng = 0.9*(self.act_mid-self.model.actuator_ctrlrange[:,0])

        #indices
        if not self.which_task==Task.MOVE_TO_LOCATION:
            self.duplicateData_switchObjs = True
            self.objInfo_start1 = self.n_jnt
            self.objInfo_start2 = self.n_jnt+3+3
            self.targetInfo_start1 = -4
            self.targetInfo_start2 = -2

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
            # self.obs_dict['object1_pos'], #3
            # self.obs_dict['object1_velp'], #3
            # self.obs_dict['object2_pos'], #3
            # self.obs_dict['object2_velp'], #3
            # self.obs_dict['target1_pos'], #2
            # self.obs_dict['target2_pos'] #2

        #get vars
        joint_pos = observations[:,:self.n_jnt]
        wrist_angle = observations[:,1]
        obj1_pos = observations[:,self.n_jnt:self.n_jnt+3]
        obj2_pos = observations[:,self.n_jnt+6:self.n_jnt+6+3]
        target1_pos = observations[:,-4:-2]
        target2_pos = observations[:,-2:]
        zeros = np.zeros(wrist_angle.shape)

        # position difference from the desired
        pos_dist_1 = np.linalg.norm(obj1_pos[:,:2] - target1_pos, axis = 1)
        pos_dist_2 = np.linalg.norm(obj2_pos[:,:2] - target2_pos, axis = 1)
        self.reward_dict['pos_dist_1'] = -pos_dist_1
        self.reward_dict['pos_dist_2'] = -pos_dist_2

        #height
        is_fall_1 = zeros
        is_fall_2 = zeros

        if not self.startup:
            height_threshold = 0.06
            is_fall_1[obj1_pos[:,2] < height_threshold] = 1
            is_fall_2[obj2_pos[:,2] < height_threshold] = 1

        if self.which_task==Task.MOVE_TO_LOCATION:
            is_fall = is_fall_1 #only care about ball #1
            self.reward_dict['drop_penalty'] = -500 * is_fall
        else:
            is_fall = np.logical_or(is_fall_1, is_fall_2) #keep both balls up
            self.reward_dict['drop_penalty'] = -500 * is_fall

        #done based on is_fall
        dones = (is_fall==1) if not self.startup else zeros

        #penalize wrist angle for lifting up (positive) too much
        wrist_threshold = 0.15
        wrist_too_high = zeros
        wrist_too_high[wrist_angle>wrist_threshold] = 1
        self.reward_dict['wrist_angle'] = -10 * wrist_too_high

        #total rewards
        if self.which_task==Task.MOVE_TO_LOCATION:
            self.reward_dict['r_total'] = 5*self.reward_dict['pos_dist_1'] + self.reward_dict['drop_penalty']
        else:
            self.reward_dict['r_total'] = 5*self.reward_dict['pos_dist_1'] + 5*self.reward_dict['pos_dist_2'] + self.reward_dict['drop_penalty'] + self.reward_dict['wrist_angle']

        #return
        if not batch_mode:
            return self.reward_dict['r_total'][0], dones[0]
        return self.reward_dict['r_total'], dones

    def get_score(self, obs):

        if self.startup:
            score = 0

        else:
            #get vars
            joint_pos = obs[:self.n_jnt]
            obj1_pos = obs[self.n_jnt:self.n_jnt+3]
            obj2_pos = obs[self.n_jnt+6:self.n_jnt+6+3]
            target1_pos = obs[-4:-2]
            target2_pos = obs[-2:]

            if self.which_task==Task.MOVE_TO_LOCATION:
                score = -np.linalg.norm(obj1_pos[:2] - target1_pos)
            else:
                pos_diff_1 = (obj1_pos[:2] - target1_pos)
                pos_diff_2 = (obj2_pos[:2] - target2_pos)
                pos_dist_1 = np.linalg.norm(pos_diff_1)
                pos_dist_2 = np.linalg.norm(pos_diff_2)
                score = -pos_dist_1-pos_dist_2

        return score


    def step(self, a):

        # set the goal as the next entry of self.goal
        if self.startup:
            if self.which_task==Task.MOVE_TO_LOCATION:
                self.desired_pose = np.array([0,0,0])
            else:
                self.desired_pose = np.array([0,0])
        else:
            self.desired_pose = self.goal[self.counter].copy()

        if self.which_task==Task.MOVE_TO_LOCATION:
            if not self.startup:
                #move the target to right location (visualization)
                self.model.site_pos[self.new_task_sid, 0] = self.desired_pose[0]
                self.model.site_pos[self.new_task_sid, 1] = self.desired_pose[1]
                self.model.site_pos[self.new_task_sid, 2] = self.desired_pose[2]+0.02

                #move the baoding targets out of the way
                self.model.site_pos[self.target1_sid, 0] = 0
                self.model.site_pos[self.target1_sid, 2] = 0.05
                self.model.site_pos[self.target2_sid, 0] = 0
                self.model.site_pos[self.target2_sid, 2] = 0.05
                self.model.site_pos[self.target1_sid, 1] = 0
                self.model.site_pos[self.target2_sid, 1] = 0

        else:
            #set target position for visualization
            desired_angle_wrt_palm = np.array([0,0])

            #shift angle by the starting ball position
            if not self.startup:
                desired_angle_wrt_palm = self.desired_pose.copy()
                desired_angle_wrt_palm[0] = desired_angle_wrt_palm[0] + self.ball_1_starting_angle
                desired_angle_wrt_palm[1] = desired_angle_wrt_palm[1] + self.ball_2_starting_angle


            #palm is x right, y down, z up
            #angle convention (world frame) is x right, y up, z out
            #(so switch our y for palm z)
            desired_positions_wrt_palm = [0,0,0,0]
            desired_positions_wrt_palm[0] = self.x_radius*np.cos(desired_angle_wrt_palm[0]) + self.center_pos[0]
            desired_positions_wrt_palm[1] = self.y_radius*np.sin(desired_angle_wrt_palm[0]) + self.center_pos[1]
            desired_positions_wrt_palm[2] = self.x_radius*np.cos(desired_angle_wrt_palm[1]) + self.center_pos[0]
            desired_positions_wrt_palm[3] = self.y_radius*np.sin(desired_angle_wrt_palm[1]) + self.center_pos[1]


            if not self.startup:
                #self.model.site_pos is in the palm frame
                #self.data.site_xpos is in the world frame (populated after a forward call)
                self.model.site_pos[self.target1_sid, 0] = desired_positions_wrt_palm[0]
                self.model.site_pos[self.target1_sid, 2] = desired_positions_wrt_palm[1]
                self.model.site_pos[self.target2_sid, 0] = desired_positions_wrt_palm[2]
                self.model.site_pos[self.target2_sid, 2] = desired_positions_wrt_palm[3]

                #move upward, to be seen
                self.model.site_pos[self.target1_sid, 1] = -0.07
                self.model.site_pos[self.target2_sid, 1] = -0.07

        # clip and scale action
        a = np.clip(a, -1.0, 1.0)
        if self.startup:
            a = a
        else:
            # mean center and scale : action = self.act_mid + a*self.act_rng
            a[a>0] = self.act_mid[a>0] + a[a>0]*self.upper_rng[a>0]
            a[a<=0] = self.act_mid[a<=0] + a[a<=0]*self.lower_rng[a<=0]

        # take the action
        self.robot.step(self, a, step_duration=self.skip*self.model.opt.timestep)
        self.counter +=1

        # get obs/rew/done/score
        obs = self._get_obs()
        reward, done = self.get_reward(obs, a)
        score = self.get_score(obs)
        env_info = {'time': self.time,
                    'obs_dict': self.obs_dict,
                    'rewards':self.reward_dict,
                    'score':score}

        return obs, reward, done, env_info

    def _get_obs(self):

        self.robot.get_obs(self, robot_noise_ratio=0, object_noise_ratio=0)
        t, qp, qv, qp_obj, qv_obj = self.robot.get_obs_from_cache(self, -1)

        self.time = t

        # hand joint positions
        self.obs_dict = {}
        self.obs_dict['robot_pos']      = qp.copy()

        # object positions
        self.obs_dict['object1_pos']     = qp_obj[:3]
        self.obs_dict['object2_pos']     = qp_obj[3+4:3+4+3]

        # object translational velocities
        self.obs_dict['object1_velp']    = qv_obj[:3]
        self.obs_dict['object2_velp']    = qv_obj[3+4:3+4+3]

        # site locations in world frame, populated after the step/forward call
        if self.which_task==Task.MOVE_TO_LOCATION:
            target_1_sid_use = self.new_task_sid
        else:
            target_1_sid_use = self.target1_sid
        self.obs_dict['target1_pos'] = np.array([self.data.site_xpos[target_1_sid_use][0], self.data.site_xpos[target_1_sid_use][1]])
        self.obs_dict['target2_pos'] = np.array([self.data.site_xpos[self.target2_sid][0], self.data.site_xpos[self.target2_sid][1]])

        return np.concatenate([ self.obs_dict['robot_pos'], #24
                        self.obs_dict['object1_pos'], #3
                        self.obs_dict['object1_velp'], #3
                        self.obs_dict['object2_pos'], #3
                        self.obs_dict['object2_velp'], #3
                        self.obs_dict['target1_pos'], #2
                        self.obs_dict['target2_pos'] #2
                        ])

    def reset_model(self):
        self.reset_pose = self.init_qpos.copy()
        self.reset_vel = self.init_qvel.copy()
        self.reset_goal = self.create_goal_trajectory()
        return self.do_reset(self.reset_pose, self.reset_vel, self.reset_goal)

    def do_reset(self, reset_pose, reset_vel, reset_goal=None):

        #### reset counters
        self.counter=0
        self.doesntSee1 = 0
        self.doesntSee2 = 0

        #### reset goal
        if reset_goal is None:
            self.goal = self.create_goal_trajectory()
        else:
            self.goal = reset_goal.copy()

        #### reset hand and objects
        self.robot.reset(self, reset_pose, reset_vel)
        self.sim.forward()
        return self._get_obs()

    def create_goal_trajectory(self):

        len_of_goals = 1000

        # populate go-to task with a target location
        if self.which_task==Task.MOVE_TO_LOCATION:
            goal_pos = np.random.randint(4)
            desired_position = []
            if goal_pos==0:
                desired_position.append(0.01)  #x
                desired_position.append(0.04)  #y
                desired_position.append(0.2)  #z
            elif goal_pos==1:
                desired_position.append(0)
                desired_position.append(-0.06)
                desired_position.append(0.24)
            elif goal_pos==2:
                desired_position.append(-0.02)
                desired_position.append(-0.02)
                desired_position.append(0.2)
            else:
                desired_position.append(0.03)
                desired_position.append(-0.02)
                desired_position.append(0.2)

            goal_traj = np.tile(desired_position, (len_of_goals, 1))

        # populate baoding task with a trajectory of goals to hit
        else:
            reward_option = 0
            period = 60
            goal_traj = []
            if self.which_task==Task.BAODING_CW:
                sign = -1
            if self.which_task==Task.BAODING_CCW:
                sign = 1

            ### Reward option: continuous circle
            if reward_option==0:
                t = 0
                while t < len_of_goals:
                    angle_before_shift = sign * 2 * np.pi * (t / period)
                    goal_traj.append(np.array([angle_before_shift, angle_before_shift]))
                    t += 1

            #### Reward option: increment in fourths
            else:
                angle_before_shift = 0
                t = 0
                while t < len_of_goals:
                    if(t>0 and t%15==0):
                        angle_before_shift += np.pi/2.0
                    goal_traj.append(np.array([angle_before_shift, angle_before_shift]))
                    t += 1

            goal_traj = np.array(goal_traj)

        return goal_traj