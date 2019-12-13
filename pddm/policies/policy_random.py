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

class Policy_Random(object):

    def __init__(self, env):

        #vars
        self.env = env
        self.low_val = -1 * np.ones(self.env.action_space.low.shape)
        self.high_val = np.ones(self.env.action_space.high.shape)
        self.shape = self.env.action_space.shape
        self.counter = 0

        self.rand_ac = np.random.uniform(self.low_val, self.high_val, self.shape)

    def get_action(self, observation, prev_action, random_sampling_params, hold_action_overrideToOne=False):

        # params for random sampling
        sample_velocities = random_sampling_params['sample_velocities']
        vel_min = random_sampling_params['vel_min']
        vel_max = random_sampling_params['vel_max']
        hold_action = random_sampling_params['hold_action']

        if hold_action_overrideToOne:
            hold_action = 1

        ### for a position-controlled robot,
        # sample random velocities
        # instead of random actions
        # (for smoother exploration)
        if sample_velocities:

            if prev_action is None:
                # generate random action for right now
                self.rand_ac = np.random.uniform(self.low_val, self.high_val, self.shape)
                action = self.rand_ac

                # generate velocity, to be used if next steps might hold_action
                self.vel_sample = np.random.uniform(vel_min, vel_max, self.env.action_space.low.shape)
                self.direction_num = np.random.randint(0, 2, self.env.action_space.low.shape)
                self.vel_sample[self.direction_num==0] = -self.vel_sample[self.direction_num==0]
            else:

                if (self.counter%hold_action)==0:
                    self.vel_sample = np.random.uniform(vel_min, vel_max, self.env.action_space.low.shape)
                    self.direction_num = np.random.randint(0, 2, self.env.action_space.low.shape)
                    self.vel_sample[self.direction_num==0] = -self.vel_sample[self.direction_num==0]

                    #go opposite direction if you hit limit
                    self.vel_sample[prev_action<=self.low_val] = np.abs(self.vel_sample)[prev_action<=self.low_val] #need to do larger action
                    self.vel_sample[prev_action>=self.high_val] = -np.abs(self.vel_sample)[prev_action>=self.high_val]
                #new action
                action = prev_action + self.vel_sample

        ### else, for a torque-controlled robot,
        # just uniformly sample random actions
        else:
            if (self.counter%hold_action)==0:
                self.rand_ac = np.random.uniform(self.low_val, self.high_val, self.shape)
            action = self.rand_ac

        self.counter +=1

        return action, 0
