import torch
import torch.nn as nn
import numpy as np
import math

init_w = 1e-3
init_b = 0.1

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class Policy(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size=256):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(in_features=obs_size, out_features=hidden_size)
        self.l2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mf = nn.Linear(in_features=hidden_size, out_features=n_actions)
        self.sf = nn.Linear(in_features=hidden_size, out_features=n_actions)
        
        self.l1.weight.data.uniform_(-init_w, init_w)
        self.l2.weight.data.uniform_(-init_w, init_w)
        self.mf.weight.data.uniform_(-init_w, init_w)
        self.sf.weight.data.uniform_(-init_w, init_w)
        self.l1.bias.data.fill_(init_b)
        self.l2.bias.data.fill_(init_b)
        self.mf.bias.data.uniform_(-init_w, init_w)
        self.sf.bias.data.uniform_(-init_w, init_w)

        self.normal = None

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        self.mu = torch.tanh(self.mf(x))
        self.sigma = torch.exp(torch.clamp(self.sf(x), LOG_SIG_MIN, LOG_SIG_MAX))
        self.normal = torch.distributions.normal.Normal(self.mu, self.sigma)
        return self.mu, self.sigma
    
    def select_action(self, x, deterministic = False):
        mu, sigma = self.forward(x)
        if deterministic:
            return mu
        else:
            return self.normal.sample()

    def log_prob(self, obs, action):
        mu, sigma = self.forward(obs)
        return self.normal.log_prob(action)

class Value(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size=256):
        super(Value, self).__init__()
        self.l1 = nn.Linear(in_features=obs_size, out_features=hidden_size)
        self.l2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.l3 = nn.Linear(in_features=hidden_size, out_features=n_actions)
        
        self.l1.weight.data.uniform_(-init_w, init_w)
        self.l2.weight.data.uniform_(-init_w, init_w)
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l1.bias.data.fill_(init_b)
        self.l2.bias.data.fill_(init_b)
        self.l3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x

class QValue(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size=256):
        super(QValue, self).__init__()
        self.l1 = nn.Linear(in_features=obs_size + n_actions, out_features=hidden_size)
        self.l2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.l3 = nn.Linear(in_features=hidden_size, out_features=n_actions)
        
        self.l1.weight.data.uniform_(-init_w, init_w)
        self.l2.weight.data.uniform_(-init_w, init_w)
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l1.bias.data.fill_(init_b)
        self.l2.bias.data.fill_(init_b)
        self.l3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x, a):
        x = torch.cat((x,a), 1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x