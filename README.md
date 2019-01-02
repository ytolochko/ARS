# ARS
This is my implementation of the Augmented Random Search algorithm as defined in the paper "Simple random search provides a competitive approach to reinforcement learning." (https://arxiv.org/abs/1803.07055) 

## Running the code
In order to run the code, you will need to have OpenAI gym, numpy and MuJoCo-py (https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key) installed.

All 4 version of ARS are implemented (V1, V1-t, V2, V2-t). In order to specify which version of the algorithm you want to use for training, please pass  the --alg argument in the comand line (default is ARS V2-t). In order to specify which environemnt you want to train it on, please pass the --env argument in the command line (default is 'HalfCheetah-v2'). 

For example, this code would run the ARS V2-t algorithm on the MuJoCo HalfCheetah environment: 

python ARS.py --alg=V2_t --env=HalfCheetah-v2

## MuJoCo and Pybullet environments. 
While MuJoCo environemnts are superior in terms of training time, there are open-source PyBullet alternatives to MuJoCo environemnts, which can also be used for training. See https://github.com/benelot/pybullet-gym for list of alternatives.
