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
import pickle
import sys
import os
import argparse
import traceback

#my imports
from pddm.utils.helper_funcs import visualize_rendering
from pddm.utils.helper_funcs import create_env
import pddm.envs

def vis_iter(args, load_dir):

    ##########################
    ## load in data
    ##########################

    #params
    paramfile = open(load_dir + '/params.pkl', 'rb')
    params = pickle.load(paramfile)
    env_name = params.env_name

    #data to visualize
    if args.eval:
        with open(load_dir + '/saved_rollouts/rollouts_eval.pickle',
                  'rb') as handle:
            rollouts_info = pickle.load(handle)
    else:
        with open(
                load_dir + '/saved_rollouts/rollouts_info_' + str(args.iter_num) +
                '.pickle', 'rb') as handle:
            rollouts_info = pickle.load(handle)

    ##########################
    ## visualize
    ##########################

    #create env
    use_env, dt_from_xml = create_env(env_name)

    rewards = []
    scores = []
    for vis_index in range(len(rollouts_info)):

        print("\n\nROLLOUT NUMBER ", vis_index, " .... num steps loaded: ", rollouts_info[vis_index]['actions'].shape[0])

        #visualize rollouts from this iteration
        _, rewards_for_rollout, scores_for_rollout = visualize_rendering(
            rollouts_info[vis_index],
            use_env,
            params,
            visualize=True,
            visualize_mpes=args.view_live_mpe_plot)
        rewards.append(np.sum(rewards_for_rollout))
        scores.append(np.mean(scores_for_rollout[-5:])) # rollout_meanFinalScore

    print("\n\n########################\nREWARDS across rollouts from this training iteration.... mean: ",
          np.mean(rewards), ", std: ", np.std(rewards))
    print("SCORES across rollouts from this training iteration.... mean: ", np.mean(scores), ", std: ",
          np.std(scores), "\n\n")


def main():
    ##########################
    ## vars to specify
    ##########################

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_path', type=str)  #address this path WRT your working directory
    parser.add_argument('--iter_num', type=int, default=1)  #if eval is False, visualize rollouts from this iteration
    parser.add_argument('--eval', action="store_true")  #if this is True, visualize rollouts from rollouts_eval.pickle
    parser.add_argument('--view_live_mpe_plot', action="store_true")
    args = parser.parse_args()

    ##########################
    ## do visualization
    ##########################

    #directory to load from
    load_dir = os.path.abspath(args.job_path)
    print("LOADING FROM: ", load_dir)
    assert os.path.isdir(load_dir)

    try:
        vis_iter(args, load_dir)
    except (KeyboardInterrupt, SystemExit):
        print('Terminating...')
        sys.exit(0)
    except Exception as e:
        print('ERROR: Exception occured while running a job....')
        traceback.print_exc()


if __name__ == '__main__':
    main()
