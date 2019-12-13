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
import copy
import matplotlib.pyplot as plt

#my imports
from pddm.samplers import trajectory_sampler
from pddm.utils.helper_funcs import do_groundtruth_rollout
from pddm.utils.helper_funcs import turn_acs_into_acsK
from pddm.utils.calculate_costs import calculate_costs
import pddm

class RandomShooting(object):

    def __init__(self, env, dyn_models, reward_func, rand_policy, use_ground_truth_dynamics,
                 execute_sideRollouts, plot_sideRollouts, params):

        ###########
        ## params
        ###########
        self.K = params.K
        self.horizon = params.horizon
        self.N = params.num_control_samples
        self.rand_policy = rand_policy
        self.use_ground_truth_dynamics = use_ground_truth_dynamics
        self.dyn_models = dyn_models
        self.execute_sideRollouts = execute_sideRollouts
        self.plot_sideRollouts = plot_sideRollouts
        self.reward_func = reward_func
        self.env = copy.deepcopy(env)
        self.random_sampling_params = dict(sample_velocities = params.rand_policy_sample_velocities,
                                        vel_min = params.rand_policy_vel_min,
                                        vel_max = params.rand_policy_vel_max,
                                        hold_action = params.rand_policy_hold_action,)

    def get_action(self, step_number, curr_state_K, actions_taken_so_far,
                   starting_fullenvstate, evaluating, take_exploratory_actions):
        """Select optimal action

        Args:
            curr_state_K:
                current "state" as known by the dynamics model
                actually a concatenation of (1) current obs, and (K-1) past obs
            step_number:
                which step number the rollout is currently on (used to calculate costs)
            actions_taken_so_far:
                used to restore state of the env to correct place,
                when using ground-truth dynamics
            starting_fullenvstate
                full state of env before this rollout, used for env resets (when using ground-truth dynamics)
            evaluating
                if True: default to not having any noise on the executing action
            take_exploratory_actions
                if True: select action based on disagreement of ensembles
                if False: (default) select action based on predicted costs

        Returns:
            best_action: optimal action to perform, according to this controller
            resulting_states_list: predicted results of executing the candidate action sequences
        """

        ############################
        ### sample N random candidate action sequences, each of length horizon
        ############################

        np.random.seed()  # to get different action samples for each rollout

        all_samples = []
        junk = 1
        for i in range(self.N):
            sample = []
            for num in range(self.horizon):
                sample.append(self.rand_policy.get_action(junk, prev_action=None, random_sampling_params=self.random_sampling_params, hold_action_overrideToOne=True)[0])
            all_samples.append(np.array(sample))
        all_samples = np.array(all_samples)

        ########################################################################
        ### make each action element be (past K actions) instead of just (curr action)
        ########################################################################

        #all_samples : [N, horizon, ac_dim]
        all_acs = turn_acs_into_acsK(actions_taken_so_far, all_samples, self.K,
                                     self.N, self.horizon)
        #all_acs : [N, horizon, K, ac_dim]

        ############################
        ### have model predict the result of executing those candidate action sequences
        ############################

        if self.use_ground_truth_dynamics:

            paths = trajectory_sampler.sample_paths_parallel(
                self.N,
                all_samples,
                actions_taken_so_far,
                starting_fullenvstate,
                self.env,
                suppress_print=True,
            )  #list of dicts, each w observations/actions/etc.

            #the taken number of paths is num_cpu*(floor(self.N/num_cpu))
            #rather than self.N, so update parameter accordingly
            self.N = len(paths)
            all_samples = all_samples[:self.N]

            resulting_states = [entry['observations'] for entry in paths]
            resulting_states = np.swapaxes(resulting_states, 0, 1)
            resulting_states_list = [resulting_states]
        else:

            resulting_states_list = self.dyn_models.do_forward_sim(
                [curr_state_K, 0], np.copy(all_acs))
            resulting_states_list = np.swapaxes(resulting_states_list, 0, 1)  #[ensSize, horizon+1, N, statesize]

        ############################
        ### evaluate the predicted trajectories
        ############################

        #calculate costs
        costs, mean_costs, std_costs = calculate_costs(resulting_states_list, all_samples,
                                self.reward_func, evaluating, take_exploratory_actions)

        #pick best action sequence
        best_score = np.min(costs)
        best_sim_number = np.argmin(costs)
        best_sequence = all_samples[best_sim_number]
        best_action = np.copy(best_sequence[0])

        #########################################
        ### execute the candidate action sequences on the real dynamics
        ### instead of just on the model
        ### useful for debugging/analysis...
        #########################################

        if self.execute_sideRollouts:
            if (step_number % self.horizon)==0:
                cmap = plt.get_cmap('jet_r')
                num_colors = 10  ##5
                indices_to_vis = [0, 1, 2]
                curr_plot = 1
                num_plots = len(indices_to_vis)
                for index_state_to_vis in indices_to_vis:
                    plt.subplot(num_plots, 1, curr_plot)
                    for sim_num in range(num_colors):
                        true_states = do_groundtruth_rollout(
                            all_samples[sim_num], self.env,
                            starting_fullenvstate, actions_taken_so_far)
                        color = cmap(float(sim_num) / num_colors)
                        if (self.iter_num == 0 and self.plot_sideRollouts):
                            if (step_number % 10 == 0):
                                plt.plot(
                                    resulting_states_list[-1]
                                    [:, sim_num, index_state_to_vis],
                                    '--',
                                    c=color,
                                    label=sim_num)
                                plt.plot(
                                    np.array(true_states)
                                    [:, index_state_to_vis],
                                    '-',
                                    c=color)
                    curr_plot += 1

                plt.legend()
                plt.show()

        return best_action, resulting_states_list
