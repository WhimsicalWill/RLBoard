import gym
import numpy as np
import wandb
from wrappers import HotStarts
from SAC.agent_class import Agent # TODO: fix broken import
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
	wandb.init('RLBoard', name='test_run_01')
	env_name = 'HalfCheetah-v3'
	save_dir = 'data'
	env = gym.make(env_name)
	env = HotStarts(env, save_dir)
	agent = Agent(alpha=0.0001, beta=0.001, input_dims=env.observation_space.shape,
					tau=0.001, batch_size=64, fc1_dims=256, fc2_dims=256, 
					action_dim=env.action_space.shape[0], env=env)
	wandb.watch([agent.actor, agent.value, agent.critic_1, agent.critic_2], log='all')


	# load the agent's saved model and the hot starts from disk
	agent.load_models()
	env.load_states()
 
	env.visualize_hot_starts(agent.choose_action)
