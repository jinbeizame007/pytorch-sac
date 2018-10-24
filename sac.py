import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import gym
import numpy as np
from copy import deepcopy
import random
import math
import csv

from models import Policy, Value, QValue
from replay_memory import ReplayMemory
from utils import soft_update, OrnsteinUhlenbeckProcess

env = gym.make("HalfCheetah-v2")
n_actions = env.action_space.shape[0]
obs_size = env.observation_space.shape[0]

policy_lr = 3e-4
vf_lr = 3e-4
qf_lr = 3e-4
gamma = 0.99
tau = 0.005
gradient_step = 1
reward_scale = 5.
warmup_step = 5000
memory_size = 1000000
batch_size = 256
memory = ReplayMemory(memory_size, batch_size, obs_size, n_actions)
ouprocess = OrnsteinUhlenbeckProcess(n_actions)

policy = Policy(obs_size, n_actions).cuda()
vf = Value(obs_size, n_actions).cuda()
qf = QValue(obs_size, n_actions).cuda()
target_vf = deepcopy(vf).cuda()

optimizer_policy = optim.Adam(policy.parameters(), lr=policy_lr)
optimizer_vf = optim.Adam(vf.parameters(), lr=vf_lr)
optimizer_qf = optim.Adam(qf.parameters(), lr=qf_lr)

policy_criterion = nn.MSELoss()
vf_criterion = nn.MSELoss()
qf_criterion = nn.MSELoss()

reward_sum = 0
episode = 0
step = 0
action = None
rewards = []

while True:
    obs = env.reset()
    obs = np.array([obs], dtype=np.float32)
    done = False

    if episode != 0:
        print("episode:",episode ," steps:", step_in_episode, " reward:", reward_sum)
    ouprocess.reset()
    reward_sum = 0
    episode += 1
    step_in_episode = 0
    done = False

    while not done:
        step += 1
        step_in_episode += 1
        
        action = policy.select_action(torch.from_numpy(obs).cuda())
        action = action.detach().cpu().numpy()[0]
        action = np.clip(action, -1, 1)

        reward = 0.
        for i in range(1):
            next_obs, reward_tmp, done, _ = env.step(action)
            next_obs = np.array([next_obs], dtype=np.float32)
            reward += reward_tmp
            reward_sum += reward_tmp
            if done:
                break

        reward *= reward_scale
        terminal = 0.
        if done:
            terminal = 1.
        memory.add(obs, action, reward, next_obs, terminal)
        obs = next_obs.copy()
            
        if step < warmup_step:
            if done:
                break
            continue
        
        for i in range(gradient_step):
            obs_batch, action_batch, reward_batch, next_obs_batch, terminal_batch = memory.sample()

            value = vf(obs_batch)
            q_value = qf(obs_batch, action_batch)
            log_prob = policy.log_prob(obs_batch, action_batch)

            ### update value function ###
            target_value = q_value - log_prob
            vf_loss = vf_criterion(value, target_value.detach())

            optimizer_vf.zero_grad()
            vf_loss.backward()
            optimizer_vf.step()

            ### update q_value function ###
            q_value = qf(obs_batch, action_batch)
            next_q_value = target_vf(next_obs_batch)
            q_target = reward_batch + (1. - terminal_batch) * gamma * next_q_value
            qf_loss = qf_criterion(q_value, q_target.detach())

            optimizer_qf.zero_grad()
            qf_loss.backward()
            optimizer_qf.step()

            ### update actor ###
            policy_loss = policy_criterion(log_prob, (q_value - value).detach())

            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            ### update target value ###
            soft_update(target_vf, vf, tau)