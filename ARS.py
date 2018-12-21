import os
import numpy as np
import gym
from gym import wrappers

# in case you don't want to use proprietary MUJOCO environemtns, there are pybullet alternatives
import pybullet_envs

class ARS():

    def __init__(self, 
                 training_length = 1000,
                 episode_length = 1000,
                 alpha = .02,
                 N_of_directions = 16,
                 b = 8,
                 noise = .03,
                 env_name = 'HalfCheetah-v2'):

        self.training_length = training_length
        self.episode_length = episode_length
        self.alpha = alpha
        self.N_of_directions = N_of_directions
        self.b = b
        self.noise = noise
        self.env_name = env_name


    def calculate_rewards(self, env, normalizer, policy, sign = None, direction = None):
        state = env.reset()
        done = False
        num_updates = 0.
        sum_rewards = 0
        while not done and num_updates < self.episode_length:
            normalizer.update(state)
            state = normalizer.normalize(state)
            action = policy.evaluate(state, sign, direction)
            state, reward, done, _ = env.step(action)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_updates += 1
        return sum_rewards

    def train(self, env, policy, normalizer):
        
        for step in range(self.training_length):
            
 g           random_directions = policy.create_random_directions()
     
            positive_rewards = [self.calculate_rewards(env, normalizer, policy, sign = '+', direction = direction) for direction in random_directions]
            negative_rewards = [self.calculate_rewards(env, normalizer, policy, sign = '-', direction = direction) for direction in random_directions]

            all_rewards = np.array(positive_rewards + negative_rewards)
            reward_sigma = np.std(all_rewards)
            
            max_rewards = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order_of_directions = sorted(max_rewards.keys(), key = lambda x:max_rewards[x], reverse = True)[:self.b]
            rollouts = [(positive_rewards[k], negative_rewards[k], random_directions[k]) for k in order_of_directions]
            
            policy.update(rollouts, reward_sigma)

            reward_evaluation = self.calculate_rewards(env, normalizer, policy)
            print('Step:', step, 'Reward:', reward_evaluation)

            # Uncomment here if you want to see how the agent prgoresses in the environment after each learning epoch
            # agent is rendered until done = True, i.e. until the episode is over (for example, it falls down)

            #state = env.reset()
            #done = False
            #while not done:
            #    env.render()
            #    action = policy.simulate_step(state)
            #    state, _, done, _ = env.step(action)

class Normalizer():
    
    def __init__(self, input_shape):
        self.n = np.zeros(input_shape)
        self.mu = np.zeros(input_shape)
        self.M2 = np.zeros(input_shape)
        self.sigma = np.zeros(input_shape)
    
    def update(self, new_state):
        # Since we do not know how long will we train, we have to keep information on the number of updates in order to do online calculation of mean and variance 
        self.n += 1
        previous_mu = self.mu # previous mu needed for online variance calculation
        self.mu += (new_state - self.mu) / self.n
        
        # Welford's Online algorithm
        delta = new_state - self.mu
        delta2 = new_state - previous_mu
        self.M2 += (delta * delta2)
        self.sigma = (self.M2/self.n).clip(min = 1e-2)

    def normalize(self, new_state):
        return (new_state - self.mu) / np.sqrt(self.sigma)


class Policy():
    
    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))
    
    def evaluate(self, state, sign = None, direction = None):
        if sign == "+":
            return (self.theta + ars.noise*direction).dot(state)
        elif sign == '-':
            return (self.theta - ars.noise*direction).dot(state)
        
        return self.theta.dot(state)
    
    def create_random_directions(self):
        return [np.random.randn(*self.theta.shape) for _ in range(ars.N_of_directions)]
    
    def update(self, rollouts, reward_sigma):
        update_step = np.zeros(self.theta.shape)
        for positive_reward, negative_reward, direction in rollouts:
            update_step += (positive_reward - negative_reward) * direction
        self.theta += ars.alpha / (ars.b * reward_sigma) * update_step

    def simulate_step(self, input):
        return self.theta.dot(input)



if __name__ == '__main__':

    env = gym.make('HalfCheetah-v2')
    #env = wrappers.Monitor(env, monitor_dir, video_callable=lambda x: True, force = True)
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]

    ars = ARS()
    policy = Policy(state_shape, action_shape)
    normalizer = Normalizer(state_shape)
    ars.train(env, policy, normalizer)