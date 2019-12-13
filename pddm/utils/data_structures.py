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

#######################


class MeanStd:

    def __init__(self):
        self.mean_x = 0
        self.mean_y = 0
        self.mean_z = 0
        self.std_x = 0
        self.std_y = 0
        self.std_z = 0


#######################


class DataPerIter:

    def __init__(self):
        self.rollouts_info = []
        self.train_rollouts_onPol = []
        self.val_rollouts_onPol = []

        self.normalization_data = MeanStd()

        self.training_losses = []
        self.training_numData = []
        self.rollouts_rewardsPerIter = []
        self.rollouts_scoresPerIter = []


#######################


class Rollout:

    def __init__(self,
                 observations=np.array([]),
                 actions=np.array([]),
                 rewards_per_step=None,
                 starting_state=None):
        self.states = observations.copy()
        self.actions = actions.copy()
        self.rewards_per_step = rewards_per_step.copy()
        self.starting_state = starting_state


#######################


class Dataset:

    def __init__(self,
                 dataX=np.array([]),
                 dataY=np.array([]),
                 dataZ=np.array([])):
        self.dataX = dataX
        self.dataY = dataY
        self.dataZ = dataZ
