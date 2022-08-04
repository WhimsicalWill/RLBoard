import gym
import numpy as np
from agent_class import Agent
from imageio import imread, mimsave

def save_vid(frames, ep_num):
	print(f"Saving gif #{ep_num}")
	mimsave(f"render-test/render-test_{ep_num}.gif", frames)

if __name__ == '__main__':
	env = gym.make('HalfCheetah-v3')
	agent = Agent(alpha=0.0001, beta=0.001, input_dims=env.observation_space.shape,
					tau=0.001, batch_size=64, fc1_dims=256, fc2_dims=256, 
					action_dim=env.action_space.shape[0], env=env)
	n_games = 10
	agent.load_models()
	for ep_num in range(n_games):
		observation = env.reset()
		done = False
		score = 0
		frames = []
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			render_img = env.render(
				mode="rgb_array",
				width=256,
				height=256,
			)
			frames.append(render_img)
			score += reward
			observation = observation_
		save_vid(frames, ep_num)
		print(f"Episode {ep_num}, score: {score}, avg_score: {avg_score}")
