# PDDM

<img src="https://github.com/google-research/pddm/blob/master/pddm/gifs/dclaw_gif.gif" height="200" /> <img src="https://github.com/google-research/pddm/blob/master/pddm/gifs/cube_gif.gif" height="200" /> <img src="https://github.com/google-research/pddm/blob/master/pddm/gifs/handwriting_gif.gif" height="200" /> <img src="https://github.com/google-research/pddm/blob/master/pddm/gifs/baoding_gif.gif" height="200" />

[[Project Page]](https://bit.ly/pddm2019) [[Paper]](https://arxiv.org/abs/1909.11652)

**Deep Dynamics Models for Learning Dexterous Manipulation**<br/>
[Anusha Nagabandi](https://people.eecs.berkeley.edu/~nagaban2/), Kurt Konolige, Sergey Levine, [Vikash Kumar](https://vikashplus.github.io/).

Please note that this is research code, and as such, is still under construction. This code implements the model-based RL algorithm presented in PDDM. Please contact Anusha Nagabandi for questions or concerns. <br/><br/>

**Contents of this README:**
- [A. Getting Started](#a-getting-started)
- [B. Quick Overview](b-quick-overview)
- [C. Train and visualize some tests](#c-train-and-visualize-some-tests)
- [D. Run experiments](#d-run-experiments)
<br/><br/>


## A. Getting started ##

#### 1) Mujoco:
Download and install mujoco (v1.5) to ~/.mujoco, following their instructions<br/>
(including setting `LD_LIBRARY_PATH` in your `~/.bashrc` file)

#### 2) If using GPU:
Setup Cuda and CUDNN verions based on your system specs.<br/>
Recommended: Cuda 8, 9, or 10.<br/>
Also, add the following to your `~/.bashrc`:
```bash
alias MJPL='LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-367/libGL.so'
```

#### 3) Setup this repo:
Without GPU support:
```bash
cd <path_to_pddm>
conda env create -f environment.yml
source activate pddm-env
pip install -e .
```

Or, for use with GPU:
```bash
cd <path_to_pddm>
conda env create -f environment_gpu.yml
source activate pddm-gpu-env
pip install -e .
```

Notes:<br/>
a) For environment_gpu to work, you'll need a working gpu and cuda/cudnn installation first.<br/>
b) Depending on your cuda/cudnn versions, you might need to change the tensorflow-gpu version specified in environment_gpu.yml. Suggestions are 1.13.1 for cuda 10, 1.12.0 for cuda 9, or 1.4.1 for cuda 8. <br/>
c) Before running any code, type the following into your terminal to activate the conda environment: <br/>
`source activate pddm-env` <br/>
d) The MJPL before the python visualization commands below are needed only if working with GPU  <br/><br/>




## B. Quick Overview ##

The overall procedure that is implemented in this code is the iterative process of learning a dynamics model and then running an MPC controller which uses that model to perform action selection. The code starts by initializing a dataset of randomly collected rollouts (i.e., collected with a random policy), and then iteratively (a) training a model on the dataset and (b) collecting rollouts (using MPC with that model) and aggregating them into the dataset.

The process of (model training + rollout collection) serves as a single iteration in this code. In other words, the rollouts from iter 0 are the result of planning under a model which was trained on randomly collected data, and the model saved at iter 3 is one that has been trained 4 times (on random data at iter 0, and on on-policy data for iters 1,2,3).

To see available parameters to set, see the files in the configs folder, as well as the list of parameters in convert_to_parser_args.py.  <br/><br/>




## C. Train and visualize some tests ##

Cheetah:
```bash
python train.py --config ../config/short_cheetah_test.txt --output_dir ../output --use_gpu
MJPL python visualize_iteration.py --job_path ../output/short_cheetah_test --iter_num 0
```

Ant:
```bash
python train.py --config ../config/short_ant_test.txt --output_dir ../output --use_gpu
MJPL python visualize_iteration.py --job_path ../output/short_ant_test --iter_num 0
```

Dclaw turn valve: <br/>
Note that this will not actually quite work, but might be reasonable.
```bash
python train.py --config ../config/short_dclaw_turn_test.txt --output_dir ../output --use_gpu
MJPL python visualize_iteration.py --job_path ../output/short_dclaw_turn_test --iter_num 0
```

Dclaw turn valve:<br/>
Note that this will work well but also take a while to run, because it's using ground-truth Mujoco dynamics for planning. It should take approximately 6 minutes on a standard laptop without any GPU.
```bash
python train.py --config ../config/test_dclaw_turn_gt.txt --output_dir ../output --use_gpu
MJPL python visualize_iteration.py --job_path ../output/dclaw_turn_gt --iter_num 0
```

Shadowhand in-hand cube rotation:<br/>
Note that this will work well but also take a while to run, because it's using ground-truth Mujoco dynamics for planning. It should take approximately 6 minutes on a standard laptop without any GPU.
```bash
python train.py --config ../config/test_cube_gt.txt --output_dir ../output --use_gpu
MJPL python visualize_iteration.py --job_path ../output/cube_gt --iter_num 0
```

Shadowhand Baoding balls:<br/>
Note that this will work well but also take a while to run, because it's using ground-truth Mujoco dynamics for planning. It should take approximately 20 minutes on a standard laptop without any GPU.
```bash
python train.py --config ../config/test_baoding_gt.txt --output_dir ../output --use_gpu
MJPL python visualize_iteration.py --job_path ../output/baoding_gt --iter_num 0
```
<br/><br/>


## D. Run experiments ##

**Train:**

```bash
python train.py --config ../config/dclaw_turn.txt --output_dir ../output --use_gpu
python train.py --config ../config/baoding.txt --output_dir ../output --use_gpu
python train.py --config ../config/cube.txt --output_dir ../output --use_gpu
```

**Evaluate a pre-trained model:**

```bash
python eval_iteration.py --job_path ../output/dclaw_turn --iter_num 0 --num_eval_rollouts 1 --eval_run_length 40
```

**Visualize:**

```bash
MJPL python visualize_iteration.py --job_path ../output/dclaw_turn --eval
MJPL python visualize_iteration.py --job_path ../output/dclaw_turn --iter_num 0
```

**Compare runs:**

Plot rewards (or scores) of multiple runs on the same plot. Note that custom labels are optional:
```bash
python compare_results.py -j ../output/runA ../output/runB -l 'mycustomlabel runA' -l 'mycustomlabel runB' --plot_rew
python compare_results.py -j ../output/runA ../output/runB -l 'mycustomlabel runA' -l 'mycustomlabel runB'
```
