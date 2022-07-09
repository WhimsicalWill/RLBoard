import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, action_dim, name,
                    chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_dim = action_dim
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = f"{chkpt_dir}/{name}_sac"

        self.fc1 = nn.Linear(input_dims[0] + action_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q1 = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device) # put the model on the device

    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        q1 = self.q1(action_value)
        
        return q1

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))

    def save_best(self):
        checkpoint_file = f"{self.chkpt_dir}/{self.name}_best"
        torch.save(self.state_dict(), checkpoint_file)

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, action_dim,
                    max_action, name, chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_dim = action_dim # this really should be 'action_dim' since it's not nec. discrete
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = f"{chkpt_dir}/{name}_sac"
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        self.mu = nn.Linear(fc2_dims, action_dim)
        self.sigma = nn.Linear(fc2_dims, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = torch.distributions.Normal(mu, sigma)

        if reparameterize: # use the reparameterization trick
            actions = probabilities.rsample() # scale and shift a standard normal -> gives differentiable sample
        else:
            actions = probabilities.sample() # sample is non-differentiable

        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise) # TODO: what is this doing
        log_probs = log_probs.sum(1, keepdim=True) # idk if this is necessary (summing over only 1 col?)

        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))
    
    def save_best(self):
        checkpoint_file = f"{self.chkpt_dir}/{self.name}_best"
        torch.save(self.state_dict(), checkpoint_file)

class ValueNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, name, chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = f"{chkpt_dir}/{name}_sac"

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.value = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))
        v = self.value(state_value)

        return v

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))
    
    def save_best(self):
        checkpoint_file = f"{self.chkpt_dir}/{self.name}_best"
        torch.save(self.state_dict(), checkpoint_file)