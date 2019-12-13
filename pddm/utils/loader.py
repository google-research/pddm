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
from pddm.utils.data_structures import DataPerIter


class Loader:

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def load_initialData(self):

        rollouts_trainRand = pickle.load(
            open(self.save_dir + '/training_data/train_rollouts_rand.pickle', 'rb'))
        rollouts_valRand = pickle.load(
            open(self.save_dir + '/training_data/val_rollouts_rand.pickle',
                 'rb'))

        return rollouts_trainRand, rollouts_valRand

    def load_iter(self, iter_num):

        data_iteration = DataPerIter()

        #info from all MPC rollouts (from this iteration)
        data_iteration.rollouts_info = pickle.load(
            open(
                self.save_dir + '/saved_rollouts/rollouts_info_' + str(iter_num)
                + '.pickle', 'rb'))

        #on-policy data (used in conjunction w random data) to train the dynamics model at this iteration
        data_iteration.train_rollouts_onPol = pickle.load(
            open(
                self.save_dir + '/training_data/train_rollouts_onPol_iter' +
                str(iter_num) + '.pickle', 'rb'))
        data_iteration.val_rollouts_onPol = pickle.load(
            open(
                self.save_dir + '/training_data/val_rollouts_onPol_iter' +
                str(iter_num) + '.pickle', 'rb'))

        #mean/std info
        data_iteration.normalization_data = pickle.load(
            open(
                self.save_dir + '/training_data/normalization_data_' +
                str(iter_num) + '.pickle', 'rb'))

        #losses/rewards/scores/sample complexity (from all iterations thus far)
        data_iteration.training_losses = np.load(
            self.save_dir +
            '/losses/list_training_loss.npy').tolist()[:iter_num]
        data_iteration.training_numData = np.load(
            self.save_dir + '/datapoints_per_agg.npy').tolist()[:iter_num]
        data_iteration.rollouts_rewardsPerIter = np.load(
            self.save_dir + '/rollouts_rewardsPerIter.npy').tolist()[:iter_num]
        data_iteration.rollouts_scoresPerIter = np.load(
            self.save_dir + '/rollouts_scoresPerIter.npy').tolist()[:iter_num]

        return data_iteration
