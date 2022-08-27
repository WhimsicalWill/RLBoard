import torch
import torch.nn.functional as F
import numpy as np
from utils import ReplayBuffer, RolloutBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork, OneStepModel

class Agent():
	def __init__(self, alpha, beta, input_dims, tau, env, action_dim,
					gamma=0.99, max_size=1000000, fc1_dims=256,
					 fc2_dims=256,  batch_size=100, reward_scale=2):
		self.gamma = gamma
		self.tau = tau
		self.batch_size = batch_size
		self.input_dims = input_dims
		self.action_dim = action_dim
		self.env = env
		self.scale = reward_scale
		self.experience_memory = ReplayBuffer(max_size, input_dims, action_dim)
		self.episode_memory = RolloutBuffer()
		
		self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, action_dim, env.action_space.high, "actor") # TODO: match params
		self.critic_1 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, action_dim, "critic_1")
		self.critic_2 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, action_dim, "critic_2")
		self.value = ValueNetwork(beta, input_dims, fc1_dims, fc2_dims, "value")
		self.target_value = ValueNetwork(beta, input_dims, fc1_dims, fc2_dims, "target_value")

		self.ensemble_size = 3
		self.hidden_size = 64
		self.curiosity_horizon = 2
		self.one_step_models = self.get_one_step_models()

		self.update_agent_parameters(tau=1) # hard update with tau=1 for initial full copying of weights

	def store_transition(self, state, action, reward, state_, done):
		self.experience_memory.store_transition(state, action, reward, state_, done)
		self.episode_memory.store_transition(state, action)

	def choose_action(self, observation):
		state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
		action, _ = self.actor.sample_normal(state, reparameterize=False)
		prediction_variance = self.get_one_step_predictions(state, action)
		# print(f"Variance of (s, a): {prediction_variance}")
		return action.cpu().detach().numpy()[0]

	def get_one_step_predictions(self, state, action):
		model_outputs = torch.zeros((len(self.one_step_models), self.input_dims[0]))
		for i in range(len(self.one_step_models)):
			model_outputs[i, :] = self.one_step_models[i](state, action)
		model_variance = torch.var(model_outputs, dim=0)
		# print(f"model_variance: {model_variance}")
		return model_variance.norm(p=2) # 2-norm of variance

	def get_one_step_models(self):
		models = []
		for i in range(self.ensemble_size):
			models.append(OneStepModel(self.input_dims, self.action_dim, self.hidden_size, f"one_step{i+1}"))
		return models

	def learn(self, obs_):
		# print(f"Learning update #{self.experience_memory.mem_ctr}")
		if self.experience_memory.mem_ctr < self.batch_size:
			return # don't learn until we can sample at least a full batch

		# Sample experience_memory buffer uniformly
		state, action, reward, state_, done = self.experience_memory.sample_buffer(self.batch_size)

		# Convert from numpy arrays to torch tensors for computation graph
		state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
		action = torch.tensor(action, dtype=torch.float).to(self.actor.device)
		state_ = torch.tensor(state_, dtype=torch.float).to(self.actor.device)
		reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
		done = torch.tensor(done, dtype=torch.float).to(self.actor.device)
		
		# value function estimates used in multiple updates
		value = self.value(state).view(-1)
		value_ = self.target_value(state_).view(-1)
		
		# <---- VALUE FUNCTION UPDATE ---->
		self.value.optimizer.zero_grad()
		actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
		log_probs = log_probs.view(-1)
		q1_new_policy = self.critic_1(state, actions)
		q2_new_policy = self.critic_2(state, actions)
		critic_value = torch.min(q1_new_policy, q2_new_policy)
		critic_value = critic_value.view(-1)
		value_target = critic_value - log_probs # State Value + Entropy in expectation
		value_loss = 0.5 * F.mse_loss(value_target, value)
		value_loss.backward(retain_graph=True)
		self.value.optimizer.step()

		# <---- ACTOR UPDATE ---->
		# Update the actor greedily w.r.t to the value function
		# We must use the reparameterization trick to get actor gradients
		self.actor.optimizer.zero_grad()
		actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
		log_probs = log_probs.view(-1) # unbatchify
		q1_new_policy = self.critic_1(state, actions)
		q2_new_policy = self.critic_2(state, actions)
		critic_value = torch.min(q1_new_policy, q2_new_policy)
		critic_value = critic_value.view(-1)
		actor_loss = log_probs - critic_value # negative of state-value + entropy
		actor_loss = torch.mean(actor_loss)
		actor_loss.backward(retain_graph=True)
		self.actor.optimizer.step()
		
		# <---- CRITIC UPDATE ---->
		self.critic_1.optimizer.zero_grad()
		self.critic_2.optimizer.zero_grad()
		q_target = self.scale*reward + (1.0 - done) * self.gamma*value_ # scale controls tradeoff
		q1_old_policy = self.critic_1(state, action).view(-1)
		q2_old_policy = self.critic_2(state, action).view(-1)
		critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_target)
		critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_target)
		(critic_1_loss + critic_2_loss).backward()
		self.critic_1.optimizer.step()
		self.critic_2.optimizer.step()

		# <---- ONE STEP MODELS UPDATE ---->
		for i in range(self.ensemble_size):
			self.one_step_models[i].optimizer.zero_grad()
			# minimize MSE between prediction of state and actual s'
			# state shape: (B, D1) | action shape: (B, D2)
			state_prediction = self.one_step_models[i](state, action)
			one_step_loss = F.mse_loss(state_prediction, state_)
			one_step_loss.backward()
			self.one_step_models[i].optimizer.step()

		# <---- STARTING DIST UPDATE ---->
		past_states = self.episode_memory.states[-self.curiosity_horizon:]
		past_actions = self.episode_memory.actions[-self.curiosity_horizon:]
		past_states = torch.tensor(past_states, dtype=torch.float32).to(self.actor.device)
		past_actions = torch.tensor(past_actions, dtype=torch.float32).to(self.actor.device)
		total_curiosity = self.calculate_curiosity(past_states, past_actions)
		starting_state = past_states[max(len(past_states) - self.curiosity_horizon, 0)]
		self.env.track_state_if_needed(total_curiosity, obs_, starting_state)

		# Do a soft update to target value function after each learning step
		self.update_agent_parameters()

	def calculate_curiosity(self, states, actions):
		total_curiosity = 0
		for i in range(states.shape[0]):
			prediction_variance = self.get_one_step_predictions(states[i], actions[i])
			total_curiosity += prediction_variance
		return total_curiosity

	def update_agent_parameters(self, tau=None):
		if tau is None:
			tau = self.tau

		for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
			updated_param = tau * param.data + (1 - tau) * target_param.data
			target_param.data.copy_(updated_param) # update the target's weights

	def save_models(self):
		print('... saving models ...')
		self.actor.save_checkpoint()
		self.critic_1.save_checkpoint()
		self.critic_2.save_checkpoint()
		self.value.save_checkpoint()
		self.target_value.save_checkpoint()

	def load_models(self):
		print('... loading models ...')
		self.actor.load_checkpoint()
		self.critic_1.load_checkpoint()
		self.critic_2.load_checkpoint()
		self.value.load_checkpoint()
		self.target_value.load_checkpoint()