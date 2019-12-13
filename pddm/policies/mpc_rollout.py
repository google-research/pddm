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
import numpy.random as npr
import time

#my imports
from pddm.policies.random_shooting import RandomShooting
from pddm.policies.cem import CEM
from pddm.policies.mppi import MPPI


class MPCRollout:

    def __init__(self,
                 env,
                 dyn_models,
                 rand_policy,
                 execute_sideRollouts,
                 plot_sideRollouts,
                 params,
                 evaluating=False):

        #init vars
        self.env = env
        self.dyn_models = dyn_models
        self.print_minimal = params.print_minimal
        self.use_ground_truth_dynamics = params.use_ground_truth_dynamics
        self.evaluating = evaluating
        self.rollout_length = params.rollout_length
        self.K = params.K
        self.rand_policy = rand_policy
        self.noise_actions = params.rollouts_noise_actions
        self.document_noised_actions = params.rollouts_document_noised_actions
        self.dt_from_xml = params.dt_from_xml
        self.noise_amount = 0.005

        self.reward_func = env.unwrapped_env.get_reward

        #init controllers
        self.controller_randshooting = RandomShooting(
                                self.env, self.dyn_models, self.reward_func, rand_policy,
                                self.use_ground_truth_dynamics,
                                execute_sideRollouts, plot_sideRollouts, params)
        self.controller_cem = CEM(self.env, self.dyn_models, self.reward_func, rand_policy,
                                  self.use_ground_truth_dynamics,
                                  execute_sideRollouts, plot_sideRollouts, params)
        self.controller_mppi = MPPI(self.env, self.dyn_models, self.reward_func, rand_policy,
                                    self.use_ground_truth_dynamics,
                                    execute_sideRollouts, plot_sideRollouts, params)

    def perform_rollout(self,
                        starting_fullenvstate,
                        starting_observation,
                        controller_type,
                        take_exploratory_actions=False,
                        isRandom=False):
        """
        Args:
            starting_fullenvstate: full state of the mujoco env (enough to allow resetting to it)
            starting_observation: obs returned by env.reset when the state itself was starting_fullenvstate
            controller_type: 'rand' or 'cem' or 'mppi'
            take_exploratory_actions: default (False) for selecting action based on optimal cost, (True) for selecting action based on ensemble disagreement
            isRandom: (True) for taking random action, default (False) for taking optimal action according to selected controller

        Populates:
            traj_taken: list of (T+1) states visited
            actions_taken: list of (T) actions taken
            total_reward_for_episode: sum of rewards for this rollout

        Returns:
            rollout_info: saving all info relevant to this rollout
        """

        rollout_start = time.time()

        #if evaluating, default to no action noise
        if self.evaluating:
            self.noise_actions = False

        #######################################
        #### select controller type
        #######################################

        if controller_type=='rand':
            get_action = self.controller_randshooting.get_action
        elif controller_type=='cem':
            get_action = self.controller_cem.get_action
        elif controller_type=='mppi':
            get_action = self.controller_mppi.get_action

        #######################################
        #### lists for saving
        #######################################

        #lists for saving info
        traj_taken = []
        traj_taken_K = []
        actions_taken = []
        actions_taken_K = []
        rewards = []
        scores = []
        env_infos = []
        list_mpe_1step = []

        #######################################
        #### init vars for rollout
        #######################################

        total_reward_for_episode = 0
        step = 0
        self.starting_fullenvstate = starting_fullenvstate

        #######################################
        #### initialize first K states/actions
        #######################################

        zero_ac = np.zeros((self.env.action_dim,))
        curr_state = np.copy(starting_observation)
        curr_state_K = [curr_state]  #not K yet, but will be

        #take (K-1) steps of action 0
        for z in range(self.K - 1):

            # take step of action 0
            curr_state, rew, _, env_info = self.env.step(zero_ac)
            step += 1

            actions_taken.append(zero_ac)
            curr_state_K.append(curr_state)

            #save info
            rewards.append(rew)
            scores.append(env_info['score'])
            env_infos.append(env_info)
            total_reward_for_episode += rew

            # Note: rewards/actions/etc. are populated during these first K steps
            # but traj_taken_K/traj_taken are not
            # because curr_state_K is not of size K yet

        #curr_state_K has K entries now
        traj_taken.append(curr_state)
        traj_taken_K.append(curr_state_K)

        #######################################
        #### loop over steps in rollout
        #######################################

        done = False
        while not(done or step>=self.rollout_length):

            if self.use_ground_truth_dynamics:
                print(step)

            ########################
            #### get optimal action
            ########################

            #curr_state_K : past K states
            #actions_taken : past all actions (taken so far in this rollout)
            if isRandom:
                best_action, _ = self.rand_policy.get_action(None, None)
            else:
                best_action, predicted_states_list = get_action(
                    step, curr_state_K, actions_taken, starting_fullenvstate,
                    self.evaluating, take_exploratory_actions)

            #noise the action, as needed
            action_to_take = np.copy(best_action)
            clean_action = np.copy(action_to_take)
            if self.noise_actions:
                noise = self.noise_amount * npr.normal(
                    size=action_to_take.shape)
                action_to_take = action_to_take + noise
                action_to_take = np.clip(action_to_take, -1, 1)
            if self.document_noised_actions:
                action_to_document = np.copy(action_to_take)
            else:
                action_to_document = np.copy(clean_action)

            ########################
            #### execute the action
            ########################
            next_state, rew, done, env_info = self.env.step(action_to_take)

            #################################################
            #### get predicted next_state
            ########## use it to calculate model prediction error (mpe)
            #################################################

            #get updated mean/std from the dynamics model
            curr_mean_x = self.dyn_models.normalization_data.mean_x
            curr_std_x = self.dyn_models.normalization_data.std_x
            next_state_preprocessed = (next_state - curr_mean_x) / curr_std_x

            #the most recent (K-1) acs
            acs_Kminus1 = np.array(actions_taken[-(self.K - 1):])  #[K-1, acDim]

            #create (past k) acs by combining (acs_Kminus1) with action
            if self.K==1:
                acs_K = np.expand_dims(action_to_document, 0)
            else:
                acs_K = np.append(acs_Kminus1,
                                  np.expand_dims(action_to_document, 0),
                                  0)  #[K, acDim]

            #Look at prediction from the 1st model of the ensemble
            predicted_next_state = self.dyn_models.do_forward_sim_singleModel(
                [curr_state_K], [acs_K])
            predicted_next_state_preprocessed = (
                predicted_next_state - curr_mean_x) / curr_std_x
            mpe_1step = np.mean(
                np.square(predicted_next_state_preprocessed -
                          next_state_preprocessed))
            list_mpe_1step.append(mpe_1step)

            ################################
            #### save things + check if done
            ################################

            #save things
            rewards.append(rew)
            scores.append(env_info['score'])
            env_infos.append(env_info)
            actions_taken.append(action_to_document)
            total_reward_for_episode += rew

            #returned by taking a step in the env
            curr_state = np.copy(next_state)

            #remove current oldest element of K list (0th entry of 0th axis)
            curr_state_K = np.delete(curr_state_K, 0, 0)
            #add this new state to end of K list
            curr_state_K = np.append(curr_state_K, np.expand_dims(
                curr_state, 0), 0)

            #save
            traj_taken.append(curr_state)
            traj_taken_K.append(curr_state_K)
            actions_taken_K.append(acs_K)

            if not self.print_minimal:
                if (step % 100 == 0):
                    print("done step ", step, ", rew: ",
                          total_reward_for_episode)

            #update
            step += 1

        ##########################
        ##### save and return
        ##########################

        if not self.print_minimal:
            print("DONE TAKING ", step, " STEPS.")
            print("Total reward: ", total_reward_for_episode)

        rollout_info = dict(
            starting_state=starting_fullenvstate,
            observations=np.array(traj_taken),
            actions=np.array(actions_taken),

            rollout_rewardsPerStep=np.array(rewards),
            rollout_rewardTotal=total_reward_for_episode,

            rollout_scoresPerStep=np.array(scores),
            rollout_meanScore=np.mean(scores),
            rollout_meanFinalScore=np.mean(scores[-5:]),

            mpe_1step=list_mpe_1step,
            observations_K=traj_taken_K,
            actions_K=actions_taken_K,
            env_infos=env_infos,
            dt_from_xml=self.dt_from_xml,
        )

        if not self.print_minimal:
            print("Time for 1 rollout: {:0.2f} s\n\n".format(time.time() - rollout_start))
        return rollout_info