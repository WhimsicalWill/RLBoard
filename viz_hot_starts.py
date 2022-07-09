import gym
import numpy as np
from agent_class import Agent

if __name__ == '__main__':
    env = gym.make('HalfCheetah-v3')
	agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, 
					input_dims=env.observation_space.shape,
					tau=0.005, batch_size=256, fc1_dims=256, fc2_dims=256, 
					env=env, action_dim=env.action_space.shape[0])
	n_games = 250
	filename = f'InvertedPendulum_scale_{agent.scale}_{n_games}_games'
	figure_file = f'plots/{filename}.png'

	best_score = env.reward_range[0] # init to smallest possible reward
	score_history = []
	load_checkpoint = False

	if load_checkpoint:
		agent.load_models()
		env.render(mode='human')