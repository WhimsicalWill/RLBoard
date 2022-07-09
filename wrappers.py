import gym
import numpy as np
import pickle

# Right now we save and load sim states to the disk
# TODO: Alternatively, we can keep a running priority queue (PQ) of sim states
# and have them actively in memory

class SimSave(gym.Wrapper):
	def __init__(self, env, filename):
		super(SimSave, self).__init__(env)
		self.filename = filename
		self.env = env

	# save the state of the simulator to a file 
	def save_state(self, checkpoint_num):
		print("saving simulator state")
		sim_state = self.env.sim.get_state()
		filename = f"{self.filename}_{checkpoint_num}.pkl"
		with open(filename, 'wb') as file:
			pickle.dump(sim_state, file, pickle.HIGHEST_PROTOCOL)

	# loads the simulator with a saved state and return current observation
	def load_state(self, checkpoint_num):
		print("loading best")
		self.env.reset()
		filename = f"{self.filename}_{checkpoint_num}.pkl"
		with open(filename, 'rb') as file:
			state_data = pickle.load(file)
			# print(state_data)
			self.env.sim.set_state(state_data)


# def make_env(env_name, shape=(42, 42, 1), repeat=4):
# 	env = gym.make(env_name)
# 	env = RepeatAction(env, repeat)
# 	env = PreprocessFrame(shape, env)
# 	env = StackFrames(env, repeat)
# 	return env