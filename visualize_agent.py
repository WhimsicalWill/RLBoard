import gym
import numpy as np
from agent_class import Agent
from hotstarts import HotStarts
from collect import collect_random_hot_starts

if __name__ == '__main__':
	env_name = 'Ant-v3'
	env = gym.make(env_name)

	# number of hot starts to render (must be a square)
	num_hot_starts = 9

	# setup HotStarts env wrapper
	saved_hot_starts_dir = 'data'
	env = HotStarts(env, saved_hot_starts_dir, num_hot_starts)

	print("Collecting random hot starts")

	collect_random_hot_starts(env, num_hot_starts)

	print(f"Collected {num_hot_starts} hot starts")

	# initialize trained agent
	agent = Agent(alpha=0.0003, beta=0.0003, 
				input_dims=env.observation_space.shape, env=env, 
				action_dim=env.action_space.shape[0], 
				fc1_dims=256, fc2_dims=256)

	# load saved model checkpoints
	agent.load_models()

	print(f"Successfully loaded models")

	# create gif of agent performing starting from each hot start
	env.visualize_hot_starts(agent.choose_action)