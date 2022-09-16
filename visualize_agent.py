import gym
import numpy as np
from SAC.agent_class import Agent
from wrappers import HotStarts

if __name__ == '__main__':
	env_name = 'Ant-v3'
	env = gym.make(env_name)

	# setup HotStarts env wrapper
	env = HotStarts(env, save_dir)
	env.load_states()

	# initialize trained agent
	agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, 
					input_dims=env.observation_space.shape,
					tau=0.005, batch_size=256, fc1_dims=256, fc2_dims=256, 
					env=env, action_dim=env.action_space.shape[0])
	agent.load_models()

	# create gif of agent performing starting from each hot start
	env.visualize_hot_starts(agent.choose_action)