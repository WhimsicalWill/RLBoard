import gym
import numpy as np
from wrappers import HotStarts

def should_save_checkpoint():
	return np.random.random() < .001

if __name__ == '__main__':
	env_name = 'HalfCheetah-v3'
	save_dir = 'data'
	env = gym.make(env_name)
	env = HotStarts(env, save_dir)
 
	n_games = 100
	for i in range(n_games):
		observation = env.reset()
		done = False
		score, steps = 0, 0
		while not done:
			action = env.action_space.sample()
			observation_, reward, done, info = env.step(action)
			if should_save_checkpoint():
				env.track_state(observation_) # save state of env as a hot start
			score += reward
			steps += 1
		
		print(f"Episode {i}, score: {score}, steps: {steps}")
	env.save_states() # save the hot starts to disk

