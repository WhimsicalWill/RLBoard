import gym
import numpy as np
from wrappers import EnvSave
from imageio import imread, mimsave

def save_vid(frames, checkpoint):
    print("Saving gif")
    mimsave(f"viz/checkpoint_{checkpoint}.gif", frames)

if __name__ == '__main__':
	env_name = 'HalfCheetah-v3'
	filename = 'data/state_data'
	env = gym.make(env_name)
	env = EnvSave(env, filename)

	max_steps = 100
	n_checkpoints = 10
	for checkpoint in range(n_checkpoints):
		env.reset() # TODO: not sure if this is needed; could embed within load_state as well
		env.load_state(checkpoint)
		done = False
		score, steps = 0, 0
		frames = []
		while not done and steps < max_steps:
			action = env.action_space.sample()
			observation_, reward, done, info = env.step(action)
			render_img = env.render(
				mode="rgb_array",
				width=256,
				height=256,
			)
			frames.append(render_img)
			score += reward
			steps += 1
		print(f"Checkpoint {checkpoint}, score: {score}, steps: {steps}")
		save_vid(frames, checkpoint)