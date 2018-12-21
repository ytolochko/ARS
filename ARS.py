"""
This is an implementation of the Augmented Random Search algorithm as described in the paper by Horia Mania, Aurelia Guy, Benjamin Recht. https://arxiv.org/pdf/1803.07055.pdf
"""

import gym 
import numpy as np

env_name = 'HalfCheetah-v2'
train_length = 1000


class Policy():

	def __init__(self, input_shape, output_shape):
		# Theta is the matrix (parameter of the policy) that is used for calculating the action proposed by the policy. See algorithm 1 and 2 in the paper
		self.theta = np.zeroes((output_shape, input_shape))

	def collect_rewards(self):
		pass

	def update(self):
		pass

def train(length):
	pass
	#for episode in range(train_length):




if __name__ == '__main__':

	env = gym.make(env_name)
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.shape[0]

	policy = Policy(state_shape, action_shape)
