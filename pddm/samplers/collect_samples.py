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
import time
import matplotlib.pyplot as plt
import copy
from functools import partial

#my imports
from pddm.utils.data_structures import *
from pddm.utils.helper_funcs import render_env

class CollectSamples(object):

    def __init__(self, env, policy, visualize_rollouts, dt_from_xml, is_random, random_sampling_params):
        self.env = env
        self.policy = policy
        self.visualize_at_all = visualize_rollouts
        self.dt_from_xml = dt_from_xml
        self.is_random = is_random
        self.random_sampling_params = random_sampling_params

    def collect_samples(self, num_rollouts, steps_per_rollout):

        visualization_frequency = 1
        rollouts = []
        for rollout_number in range(num_rollouts):

            observation, starting_state = self.env.reset(return_start_state=True)

            rollout = self.perform_rollout(
                observation, steps_per_rollout, rollout_number,
                visualization_frequency, starting_state)

            rollouts.append(rollout)

        return rollouts

    def perform_rollout(self, observation, steps_per_rollout, rollout_number,
                        visualization_frequency, starting_state):
        observations = []
        actions = []
        visualize = False
        rewards_per_step = []
        if ((rollout_number % visualization_frequency) == 0):
            print("Currently performing rollout #", rollout_number)
            if (self.visualize_at_all):
                print("---- visualizing a rollout ----")
                visualize = True

        starttime = time.time()
        prev_action = None
        for step_num in range(steps_per_rollout):

            # get action
            if self.is_random:
                action, _ = self.policy.get_action(observation, prev_action, self.random_sampling_params)
            else:
                action, _ = self.policy.get_action(observation)

            # store things
            observations.append(observation)
            actions.append(action)
            prev_action = action.copy()

            # take step
            next_observation, reward, terminal, _ = self.env.step(action)

            # store things
            rewards_per_step.append(reward)
            observation = np.copy(next_observation)

            if terminal:
                print("Had to stop rollout because terminal state was reached.")
                break

            if visualize:
                render_env(self.env) #this render being on slows down the timing below, but that's ok

        if visualize:
            print("    1 rollout, time taken: {:0.4f} s".format(time.time() - starttime))
            print("dt: ", self.dt_from_xml)
            print("num steps: ", steps_per_rollout)
            print("should be: ", self.dt_from_xml*steps_per_rollout)
        return Rollout(
            np.array(observations), np.array(actions),
            np.array(rewards_per_step), starting_state)
