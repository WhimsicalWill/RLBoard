import torch
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env, action_dim, 
                    gamma=0.99, max_size=1000000, fc1_dims=256,
                     fc2_dims=256,  batch_size=100, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.scale = reward_scale
        self.memory = ReplayBuffer(max_size, input_dims, action_dim)
        
        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, action_dim, env.action_space.high, "actor") # TODO: match params
        self.critic_1 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, action_dim, "critic_1")
        self.critic_2 = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, action_dim, "critic_2")
        self.value = ValueNetwork(beta, input_dims, fc1_dims, fc2_dims, "value")
        self.target_value = ValueNetwork(beta, input_dims, fc1_dims, fc2_dims, "target_value")

        self.update_agent_parameters(tau=1) # hard update with tau=1 for initial full copying of weights

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return # don't learn until we can sample at least a full batch

        # Sample memory buffer uniformly
        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

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

        # Do a soft update to target value function after each learning step
        self.update_agent_parameters()

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
