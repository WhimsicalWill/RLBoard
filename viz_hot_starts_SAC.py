import gym
import numpy as np
from wrappers import EnvSave
from imageio import imread, mimsave
from SAC.agent_class import Agent # TODO: fix broken import

def save_vid(frames, checkpoint):
	print("Saving gif")
	mimsave(f"viz/checkpoint_{checkpoint}.gif", frames)

if __name__ == '__main__':
	env_name = 'HalfCheetah-v3'
	filename = 'data/state_data'
	env = gym.make(env_name)
	env = EnvSave(env, filename)
	agent = Agent(alpha=0.0001, beta=0.001, input_dims=env.observation_space.shape,
					tau=0.001, batch_size=64, fc1_dims=256, fc2_dims=256, 
					action_dim=env.action_space.shape[0], env=env)

	# load the agent's saved model
	agent.load_models()

	max_steps = 500
	n_checkpoints = 10
	for checkpoint in range(n_checkpoints):
		obs = env.load_state(checkpoint)
		done = False
		score, steps = 0, 0
		frames = []
		while not done and steps < max_steps:
			action = agent.choose_action(obs)
			obs_, reward, done, info = env.step(action)
			render_img = env.render(
				mode="rgb_array",
				width=256,
				height=256,
			)
			frames.append(render_img)
			score += reward
			steps += 1
			obs = obs_
		print(f"Checkpoint {checkpoint}, score: {score}, steps: {steps}")
		save_vid(frames, checkpoint)