import gym
import numpy as np
from wrappers import SimSave

def should_save_checkpoint():
	return np.random.random() < .001

if __name__ == '__main__':
	env_name = 'HalfCheetah-v3'
	filename = 'data/state_data'
	env = gym.make(env_name)
	env = SimSave(env, filename)
 
	n_games = 100
	num_checkpoints = 10
	checkpoint = 0	
 
	for i in range(n_games):
		observation = env.reset()
		done = False
		score, steps = 0, 0
		while not done:
			action = env.action_space.sample()
			observation_, reward, done, info = env.step(action)
			if should_save_checkpoint() and checkpoint < num_checkpoints:
				env.save_state(observation_, checkpoint) # TODO: save state and current obs
				checkpoint += 1
			score += reward
			steps += 1
		
		print(f"Episode {i}, score: {score}, steps: {steps}")

