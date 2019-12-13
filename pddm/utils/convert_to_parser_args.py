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

import sys
import argparse

def convert_to_parser_args(args_source=sys.argv[1:]):

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)

    #######################
    ### experiment info
    #######################

    parser.add_argument('--env_name', type=str)
    parser.add_argument('--rollout_length', type=int)
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--num_trajectories_per_iter', type=int, default=2)

    # -1 means start from scratch... any other number says which iter to restore & continue from
    parser.add_argument('--continue_run', type=int, default=-1)
    parser.add_argument('--continue_run_filepath', type=str, default='')

    # have controller use true dynamics for planning (instead of learned model)
    parser.add_argument('--use_ground_truth_dynamics', action="store_true")

    # other T/F
    parser.add_argument('--visualize_MPC_rollout', action="store_true")
    parser.add_argument('--print_minimal', action="store_true")

    # noise options
    parser.add_argument('--make_aggregated_dataset_noisy', action="store_true")
    parser.add_argument('--make_training_dataset_noisy', action="store_true")
    parser.add_argument('--rollouts_noise_actions', action="store_true")
    parser.add_argument('--rollouts_document_noised_actions', action="store_true")

    ###########################
    ### random data collection
    ###########################

    # collect random rollouts
    parser.add_argument('--load_existing_random_data', action="store_true")
    parser.add_argument('--num_rand_rollouts_train', type=int, default=100)
    parser.add_argument('--num_rand_rollouts_val', type=int, default=50)
    parser.add_argument('--rand_rollout_length', type=int, default=30)
    parser.add_argument('--use_threading', action="store_true")

    # sample random velocities vs. positions
    parser.add_argument('--rand_policy_sample_velocities', action="store_true")
    parser.add_argument('--rand_policy_vel_min', type=float, default=0)
    parser.add_argument('--rand_policy_vel_max', type=float, default=0)
    parser.add_argument('--rand_policy_hold_action', type=int, default=1)

    #######################
    ### dynamics model
    #######################

    # arch
    parser.add_argument('--num_fc_layers', type=int, default=2)
    parser.add_argument('--depth_fc_layers', type=int, default=64)
    parser.add_argument('--ensemble_size', type=int, default=1) #ensemble size
    parser.add_argument('--K', type=int, default=1) #number of past states for input to model

    # False to start model training from SCRATCH at each iteration
    parser.add_argument('--warmstart_training', action="store_true")

    # model training
    parser.add_argument('--always_use_savedModel', action="store_true") #use saved model instead of training one
    parser.add_argument('--batchsize', type=int, default=500) #batchsize for each gradient step
    parser.add_argument('--lr', type=float, default=0.001) #learning rate
    parser.add_argument('--nEpoch', type=int, default=40) #epochs of training
    parser.add_argument('--nEpoch_init', type=int, default=40) #epochs of training for 1st iter

    #######################
    ### controller
    #######################

    # MPC
    parser.add_argument('--horizon', type=int, default=7) #planning horizon
    parser.add_argument('--num_control_samples', type=int, default=700) #number of candidate ac sequences
    parser.add_argument('--controller_type', type=str, default='mppi') #rand, cem, mppi

    # cem
    parser.add_argument('--cem_max_iters', type=int, default=3) #number of iters
    parser.add_argument('--cem_num_elites', type=int, default=20) #elites for refitting sampling dist

    # mppi
    parser.add_argument('--mppi_kappa', type=float, default=1.0) #reward weighting
    parser.add_argument('--mppi_mag_noise', type=float, default=0.9) #magnitude of sampled noise
    parser.add_argument('--mppi_beta', type=float, default=0.8) #smoothing

    args = parser.parse_args(args_source)
    return args
