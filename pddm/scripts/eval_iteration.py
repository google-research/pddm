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
import numpy.random as npr
import os
import sys
import argparse
import tensorflow as tf
import traceback

#my imports
from pddm.utils.helper_funcs import create_env
from pddm.utils.helper_funcs import get_gpu_config
from pddm.policies.policy_random import Policy_Random
from pddm.utils.loader import Loader
from pddm.regressors.dynamics_model import Dyn_Model
from pddm.policies.mpc_rollout import MPCRollout
from pddm.utils.data_structures import *
import pddm.envs

def run_eval(args, save_dir):

    ##########################
    ## params
    ##########################

    ### read in params from saved config file
    paramfile = open(save_dir + '/params.pkl', 'rb')
    params = pickle.load(paramfile)

    ### can manually set some options here, for these eval runs (to override options from training)
    # params.kappa = 1
    # params.horizon = 20
    # params.mppi_beta = 0.6

    #overwrite config's value with the commandline arg value
    params.use_ground_truth_dynamics = args.use_ground_truth_dynamics

    #if run length wasn't specified in args, default to config file's value
    if args.eval_run_length==-1:
        args.eval_run_length = params.rollout_length

    ##########################
    ## other initializations
    ##########################

    ### set seeds
    npr.seed(args.seed)
    tf.set_random_seed(args.seed)

    #loader and data processor
    loader = Loader(save_dir)

    #env, rand policy
    env, dt_from_xml = create_env(params.env_name)
    random_policy = Policy_Random(env.env)

    #load data from the iteration (for plotting)
    iter_data = loader.load_iter(args.iter_num)
    trainingLoss_perIter = iter_data.training_losses
    rew_perIter = iter_data.rollouts_rewardsPerIter
    scores_perIter = iter_data.rollouts_scoresPerIter
    trainingData_perIter = iter_data.training_numData

    #mean/std info
    normalization_data = iter_data.normalization_data

    ### data dims
    outputSize = normalization_data.mean_z.shape[0]
    acSize = normalization_data.mean_y.shape[0]
    inputSize = normalization_data.mean_x.shape[0] + acSize

    with tf.Session(config=get_gpu_config(args.use_gpu, args.gpu_frac)) as sess:

        ##############################################
        ### dynamics model + controller
        ##############################################

        dyn_models = Dyn_Model(inputSize, outputSize, acSize, sess, params=params)

        mpc_rollout = MPCRollout(
            env,
            dyn_models,
            random_policy,
            execute_sideRollouts=args.execute_sideRollouts,
            plot_sideRollouts=True,
            params=params)

        ##############################################
        ### restore the saved dynamics model
        ##############################################

        #restore model
        sess.run(tf.global_variables_initializer())
        restore_path = save_dir + '/models/model_aggIter' + str(
            args.iter_num) + '.ckpt'
        saver = tf.train.Saver(max_to_keep=0)
        saver.restore(sess, restore_path)
        print("\n\nModel restored from ", restore_path, "\n\n")

        #restore mean/std
        dyn_models.normalization_data = normalization_data

        ################################
        ########### RUN ROLLOUTS
        ################################

        list_rewards = []
        list_scores = []
        rollouts = []

        for rollout_num in range(args.num_eval_rollouts):

            # Note: if you want to evaluate a particular goal, call env.reset with a reset_state
            # where that reset_state dict has reset_pose, reset_vel, and reset_goal
            starting_observation, starting_state = env.reset(return_start_state=True)

            if not params.print_minimal:
                print("\n############# Performing MPC rollout #", rollout_num)

            mpc_rollout.rollout_length = args.eval_run_length
            rollout_info = mpc_rollout.perform_rollout(
                starting_state,
                starting_observation,
                controller_type=params.controller_type,
                take_exploratory_actions=False)

            #save info from MPC rollout
            list_rewards.append(rollout_info['rollout_rewardTotal'])
            list_scores.append(rollout_info['rollout_meanFinalScore'])
            rollouts.append(rollout_info)

        #save all eval rollouts
        pickle.dump(
            rollouts,
            open(save_dir + '/saved_rollouts/rollouts_eval.pickle', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)
        print("REWARDS: ", list_rewards, " .... mean: ", np.mean(list_rewards), " std: ", np.std(list_rewards))
        print("SCORES: ", list_scores, " ... mean: ", np.mean(list_scores), " std: ", np.std(list_scores), "\n\n")


def main():

    #############################
    ## vars to specify for eval
    #############################

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job_path', type=str,
        default='../output/cheetah')  #address this WRT working directory
    parser.add_argument('--iter_num', type=int, default=0)
    parser.add_argument('--num_eval_rollouts', type=int, default=3)
    parser.add_argument('--eval_run_length', type=int, default=-1)
    parser.add_argument('--gpu_frac', type=float, default=0.9)
    parser.add_argument('--use_ground_truth_dynamics', action="store_true")
    parser.add_argument('--execute_sideRollouts', action="store_true")
    parser.add_argument('--use_gpu', action="store_true")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    #directory to load from
    save_dir = os.path.abspath(args.job_path)
    assert os.path.isdir(save_dir)

    ##########################
    ## run evaluation
    ##########################

    try:
        run_eval(args, save_dir)
    except (KeyboardInterrupt, SystemExit):
        print('Terminating...')
        sys.exit(0)
    except Exception as e:
        print('ERROR: Exception occured while running a job....')
        traceback.print_exc()


if __name__ == '__main__':
    main()
