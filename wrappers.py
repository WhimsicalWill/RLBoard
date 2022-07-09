import gym
import numpy as np
import pickle
import os

class EnvState():
	def __init__(self, sim_state, first_obs, priority=0):
		self.sim_state = sim_state
		self.first_obs = first_obs
		self.priority = priority

class EnvSave(gym.Wrapper):
	def __init__(self, env, filename):
		super(EnvSave, self).__init__(env)
		self.filename = filename
		self.env = env
		self.hot_starts = [] # to be populated by user with starting env states

	# save the state of the env to a file 
	def save_state(self, obs_, checkpoint_num):
		print("saving env state")
		filename = f"{self.filename}_{checkpoint_num}.pkl"

		sim_data = self.env.sim.get_state()
		env_state = EnvState(sim_data, obs_)
		with open(filename, 'wb') as file:
			pickle.dump(env_state, file, pickle.HIGHEST_PROTOCOL)

	# loads the env with a saved state and return current observation
	def load_state(self, checkpoint_num):
		print("loading env state")
		filename = f"{self.filename}_{checkpoint_num}.pkl"

		with open(filename, 'rb') as file:
			env_state = pickle.load(file)
			self.env.sim.set_state(env_state.env_state)
			return env_state.first_obs

	def load_states(self, dir):
		for filename in os.listdir(dir):
			with open(filename, 'rb') as file:
				env_state = pickle.load(file)
				self.hot_starts.append(env_state)

# Right now we save and load sim states to the disk
# TODO: This env should track a PQ of sim states and have them actively in memory
# with some maximum buffer size. 
# The amount of data in a sim state may differ by env.
# This file should also include a rudimentary class for a Sim State, which includes
# a priority score, the sim data, and the first observation.
#
#
