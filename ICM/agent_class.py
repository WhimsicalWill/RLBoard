import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from memory import Memory
from networks import ActorCritic
from ICM import ICM

class AgentProcess():
    def __init__(self, input_shape, n_actions, global_ac=None, 
                ac_optimizer=None, global_icm=None, icm_optimizer=None,
                gamma=0.99, tau=1.0, beta=0.2, alpha=0.1):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.actor_critic = ActorCritic(input_shape, n_actions)

        if ac_optimizer is not None:
            self.global_ac = global_ac # the shared global controller
            self.ac_optimizer = ac_optimizer
            self.global_icm = global_icm
            self.icm_optimizer = icm_optimizer
            self.icm = ICM(input_shape, n_actions) # not needed for evaluating a static policy
            self.memory = Memory()

    def store_transition(self, reward, value, log_prob, state, new_state, action):
        self.memory.store_transition(reward, value, log_prob, state, new_state, action)
    
    # choose action according to policy, without any eps-greedy exploration
    def choose_action(self, obs):
        state = torch.tensor([obs], dtype=torch.float) # batchify
        probs, value = self.actor_critic(state)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # no need for .cpu() before .numpy() as we are on cpu already
        return action.numpy()[0], value, log_prob

    def learn(self, obs, done):
        # load environment transitions that are used for gradient update
        rewards, values, log_probs, states, new_states, actions = self.memory.sample_memory()

        # TODO: add loss functions for ICM
        self.icm_optimizer.zero_grad()
        intrinsic_rewards, icm_loss = self.calc_icm_loss(states, new_states, actions)
        icm_loss.backward()
        self.copy_gradients_and_step(self.icm, self.global_icm, self.icm_optimizer)

        self.ac_optimizer.zero_grad()
        loss = self.calc_ac_loss(obs, done, rewards, intrinsic_rewards, values, log_probs)
        loss.backward() # compute gradient of loss w.r.t. local agent's parameters
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 40) # in-place gradient norm clip
        self.copy_gradients_and_step(self.actor_critic, self.global_ac, self.ac_optimizer) # take gradient step for global controller, and update local params

        self.memory.reset() # clear the memory after a gradient update

    # calculate the loss according to the A3C algorithm
    def calc_ac_loss(self, new_state, done, rewards, intrinsic_rewards, values, log_probs):
        rewards += intrinsic_rewards.detach().numpy() # agent's reward is sum of extrinsic and intrinsic rewards
        returns = self.calc_R(done, rewards, values)

        # if this transition is terminal, value is zero
        next_v = torch.zeros(1, 1) if done else self.actor_critic(torch.tensor([new_state], dtype=torch.float))[1]
        values.append(next_v.detach()) # detach from computation graph since it was just computed
        values = torch.cat(values).squeeze()
        log_probs = torch.cat(log_probs)
        rewards = torch.tensor(rewards)

        delta_t = rewards + self.gamma * values[1:] - values[:-1]
        n_steps = len(delta_t)
        batch_gae = np.zeros(n_steps) # initialize zero vector

        # O(n) time complexity implementation
        batch_gae[-1] = delta_t[-1]
        for t in reversed(range(n_steps - 1)):
            batch_gae[t] = delta_t[t] + (self.gamma*self.tau) * batch_gae[t+1] # TODO: why do we use gamma twice effectively?
        batch_gae = torch.tensor(batch_gae, dtype=torch.float)

        # sum works better empirically (gradient gets scaled by batch_size)
        actor_loss = -torch.sum(log_probs * batch_gae)
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)
        entropy_loss = torch.sum(log_probs * torch.exp(log_probs)) # minimize negative entropy (maximizes entropy)

        total_loss = actor_loss + critic_loss + 0.04 * entropy_loss
        return total_loss

    def calc_R(self, done, rewards, values):
        values = torch.cat(values).squeeze() # transform list to tensor

        # initialize the reward for calculating batch returns
        if len(values.size()) == 1: # batch of states
            R = values[-1] * (1 - int(done))
        elif len(values.size()) == 0:
            R = values * (1 - int(done))

        # iterate backwards over batch transitions
        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float).reshape(values.shape)

        return batch_return

    def calc_icm_loss(self, states, new_states, actions):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        new_states = torch.tensor(new_states, dtype=torch.float)

        phi_new, pi_logits, phi_hat_new = self.icm(states, new_states, actions)
        actions = actions.long() # convert FloatTensor to LongTensor for CrossEntropy loss

        cross_entropy_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        inverse_loss = (1 - self.beta) * cross_entropy_loss(pi_logits, actions)
        forward_loss = self.beta * mse_loss(phi_hat_new, phi_new)
        intrinsic_rewards = self.alpha*0.5*torch.mean((phi_hat_new - phi_new) ** 2, dim=1)
        return intrinsic_rewards, inverse_loss + forward_loss

    def copy_gradients_and_step(self, local_model, global_model, global_optimizer):
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            global_param.grad = local_param.grad
        global_optimizer.step() # update the central actor_critic with the gradients of the local model
        local_model.load_state_dict(global_model.state_dict())

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_checkpoint()

    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_checkpoint()