import torch
import numpy as np
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
	def __init__(self, alpha, beta, input_dims, env, 
				action_dim, fc1_dims=256, fc2_dims=256):
		self.action_dim = action_dim
		
		self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, action_dim, env.action_space.high, "actor")
		self.critic_1 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, action_dim, "critic_1")
		self.critic_2 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, action_dim, "critic_2")
		self.value = ValueNetwork(beta, input_dims, fc1_dims, fc2_dims, "value")
		self.target_value = ValueNetwork(beta, input_dims, fc1_dims, fc2_dims, "target_value")

	def choose_action(self, observation):
		state = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.actor.device)
		actions, _ = self.actor.sample_normal(state, reparameterize=False)
		return actions.cpu().detach().numpy()[0]

	def load_models(self):
		print('... loading models ...')
		self.actor.load_checkpoint()
		self.critic_1.load_checkpoint()
		self.critic_2.load_checkpoint()
		self.value.load_checkpoint()
		self.target_value.load_checkpoint()
