import gym
import numpy as np
from wrappers import SimSave

if __name__ == '__main__':
	env_name = 'HalfCheetah-v3'
	filename = 'data/state_data'
	env = gym.make(env_name)
	env = SimSave(env, filename)

	n_checkpoints = 10
	for checkpoint in range(n_checkpoints):
		env.reset() # TODO: not sure if this is needed; could embed within load_state as well
		env.load_state(checkpoint)
		done = False
		score, steps = 0, 0
		while not done:
			action = env.action_space.sample()
			observation_, reward, done, info = env.step(action)
			env.render()
			score += reward
			steps += 1
		print(f"Checkpoint {checkpoint}, score: {score}, steps: {steps}")