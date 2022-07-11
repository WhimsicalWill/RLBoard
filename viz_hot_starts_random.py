import gym
import numpy as np
from wrappers import HotStarts

if __name__ == '__main__':
	env_name = 'HalfCheetah-v3'
	save_dir = 'data'
	env = gym.make(env_name)
	env = HotStarts(env, save_dir)

	# random mapping from obs -> actions
	def random_action(observation):
		return env.action_space.sample()
		

	# load the hot starts from disk
	env.load_states()
	env.visualize_hot_starts(random_action)
