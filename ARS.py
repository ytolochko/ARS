"""
This is an implementation of the Augmented Random Search algorithm as described in the paper by Horia Mania, Aurelia Guy, Benjamin Recht. https://arxiv.org/pdf/1803.07055.pdf
"""

import gym 
import numpy as np
import os 
from gym import wrappers
import pybullet_envs

env_name = 'HalfCheetahBulletEnv-v0'

class Normalizer():
	def __init__(self, input_shape):
		self.n = np.zeros(input_shape)
		self.mu = np.zeros(input_shape)
		self.sigma = np.zeros(input_shape)
		self.M2 = np.zeros(input_shape)

	def update(self, new_state):
		# Since we do not know how long will we train, we have to keep information on the number of updates in order to do online calculation of mean and variance 
		self.n += 1
		previous_mu = self.mu # previous mu needed for online variance calculation
		self.mu += (new_state - self.mu) / n
		
		# Welford's Online algorithm
		delta = new_state - self.mu
		delta2 = new_state - previous_mu
		self.M2 += (delta * delta2)
		self.sigma += M2/n

	def normalize(self, new_state):
		return (new_state - self.mu) / np.sqrt(self.sigma)

class Policy():

	def __init__(self, input_shape, output_shape):
		# Theta is the matrix (parameter of the policy) that is used for calculating the action proposed by the policy. See algorithm 1 and 2 in the paper
		self.theta = np.zeros((output_shape, input_shape))

	def calculate_action(self, state, sign = None, direction = None):
		if sign == '+':
			# noise = zero mean Gaussian vector
			return  (self.theta + ars.noise * direction).dot(state)
		elif sign == '-':
			return (self.theta - ars.noise * direction).dot(state)

		return self.theta.dot(state)
	
	def update(self, rollouts, reward_sigma):
		alpha = ars.alpha
		b = ars.b
		update_step = 0
		for positive_reward, negative_reward, direction in rollouts[:b]:
			update_step += (positive_reward - negative_reward) * direction
			self.theta += alpha/(b * reward_sigma) * update_step

		


class ARS():
	def __init__(self, env,train_length = 1000, alpha = 0.02, N_of_directions = 16, noise = 0.03, N_top_performing_directions = 16):
		self.env = env
		self.state_shape = env.observation_space.shape[0]
		self.action_shape = env.action_space.shape[0]

		# Hyperparameters
		self.train_length = train_length
		self.alpha = alpha
		self.N_of_directions = N_of_directions
		self.noise = noise
		self.b = N_top_performing_directions

	def calculate_reward(self, sign, policy, direction):
		state = self.env.reset()
		sum_rewards = 0
		num_plays = 0
		done = False
		while not done and num_plays <= 1000:
			action = policy.calculate_action(state, sign, direction)
			state, reward, done, _ = self.env.step(action)
			sum_rewards += reward
			num_plays += 1
		return reward

	def evaluate_policy(self, policy):
		state = self.env.reset()
		action = policy.calculate_action(state)
		state, reward, done, _ = env.step(action)
		return reward

	def train(self, policy):
		for episode in range(self.train_length):

			random_directions = [np.random.rand(self.action_shape, self.state_shape) for _ in range(self.N_of_directions)]

			positive_rewards = [0 for _ in range(self.N_of_directions)]
			negative_rewards = [0 for _ in range(self.N_of_directions)]

			positive_rewards = [self.calculate_reward('positive', policy, direction) for direction in random_directions]
			#print('calculated positive')
			negative_rewards = [self.calculate_reward('negative', policy, direction) for direction in random_directions]
			#print('calculated negative')

			all_rewards = np.array(positive_rewards + negative_rewards)
			reward_sigma = np.std(all_rewards)

			max_rewards = {k:max(positive_reward, negative_reward) for k, (positive_reward, negative_reward) in enumerate(zip(positive_rewards, negative_rewards))}
			sorted_max_rewards_keys = sorted(max_rewards.keys(), key = lambda x : max_rewards[x])
			rollouts = [(positive_rewards[k], negative_rewards[k], random_directions[k]) for k in sorted_max_rewards_keys]
			policy.update(rollouts, reward_sigma)

			reward = self.evaluate_policy(policy)
			print('EPISODE NUMBER', episode, 'Reward:', reward)
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

if __name__ == '__main__':
	work_dir = mkdir('exp', 'brs')
	monitor_dir = mkdir(work_dir, 'monitor')

	np.random.seed(1)

	env = gym.make(env_name)
	#env = wrappers.Monitor(env, monitor_dir, force = True)
	
	ars = ARS(env)
	policy = Policy(ars.state_shape, ars.action_shape)
	normalizer = Normalizer(ars.state_shape)

	ars.train(policy)

