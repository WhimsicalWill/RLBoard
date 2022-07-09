import gym
import numpy as np

class SimSaveWrapper(gym.Wrapper):
	def __init__(self, env, filename):
		super(SimSaveWrapper, self).__init__(env)
		self.filename = filename

	def save_state(self, checkpoint):
		print("saving simulator state")
		filename = f"data/state_data_{checkpoint}.pkl"
		with open(filename, 'wb') as file:
			pickle.dump(sim_state, file, pickle.HIGHEST_PROTOCOL)

	def load_state(self, checkpoint_num):
		print("loading best")
		filename = f"data/state_data_{checkpoint}.pkl"
		with open(filename, 'rb') as file:
			state_data = pickle.load(file)
			print(state_data)


# def make_env(env_name, shape=(42, 42, 1), repeat=4):
# 	env = gym.make(env_name)
# 	env = RepeatAction(env, repeat)
# 	env = PreprocessFrame(shape, env)
# 	env = StackFrames(env, repeat)
# 	return env