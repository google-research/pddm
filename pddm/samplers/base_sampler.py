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

import logging
logging.disable(logging.CRITICAL)
import numpy as np
import copy


# Single core rollout to sample trajectories

def do_rollout(N_percpu,
               actions_list,
               actions_taken_so_far,
               starting_fullenvstate,
               which_cpu,
               env=None,):
    """
    N_percpu        : number of trajectories to execute on this cpu
    actions_list    : list of ALL candidate action sequences
    actions_taken_so_far : all actions taken so far (to help reset the env to the right place)
    which_cpu       : which cpu this is
    env             : env object to sample from
    """

    T = len(actions_list[0])  #horizon
    paths = []
    for ep in range(N_percpu):

        sim_num = (which_cpu * N_percpu) + ep

        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []

        #reset env to what it was at the beginning of the original MPC rollout
        curr_env = copy.deepcopy(env)
        o = curr_env.reset(reset_state=starting_fullenvstate)

        step = 0
        #take the steps taken so far, so this forward prediction is done from the correct state
        for ac in actions_taken_so_far:
            if ac.shape[0]==1:
                ac = ac[0]
            o, _, _, _ = curr_env.step(ac)
            step += 1

        #take the steps of this candidate action sequence
        for t in range(T):

            #action to execute
            a = actions_list[sim_num][t]

            next_o, r, done, env_info = curr_env.step(a)

            #save info
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(0)
            env_infos.append(0) #env_info #causes error when env_info is a dictionary
            o = next_o

            step += 1

        #save all results of this candidate action sequence
        observations.append(o)
        path = dict(
            observations=np.array(observations),
            # ^ starts w starting state, so 1 more entry than the others
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=0,
            env_infos=env_infos,
            terminated=done)
        paths.append(path)

    return paths


def do_rollout_star(args_list):
    return do_rollout(*args_list)
