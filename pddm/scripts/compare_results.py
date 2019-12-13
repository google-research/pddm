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
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import traceback

#my imports
from pddm.utils.helper_funcs import plot_mean_std

def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', action='append', nargs='+', help='job/experiment')
    parser.add_argument('-l', '--labels', action='append', nargs='?', help='label for plotting that experiment')
    parser.add_argument('-plot_rew', '--plot_rew', action='store_true')
    args = parser.parse_args()
    jobs = args.jobs[0]

    # scan labels
    if args.labels is not None:
        assert (len(jobs)==len(args.labels)), "The number of labels has to be same as the number of jobs"
    else:
        args.labels = ['']*len(jobs)

    # Scan jobs and plot
    colors=['r', 'g', 'b', 'k', 'c', 'm', 'pink', 'purple']
    for i in range(len(jobs)):
        if args.plot_rew:
            print("LOOKING AT REW")
            rew = np.load(jobs[i] + '/rollouts_rewardsPerIter.npy')
        else:
            print("LOOKING AT SCORE")
            rew = np.load(jobs[i] + '/rollouts_scoresPerIter.npy')
        plot_mean_std(rew[:, 0], rew[:, 1], label=args.labels[i], newfig=False, color=colors[i])

    plt.show()

if __name__ == '__main__':
    main()