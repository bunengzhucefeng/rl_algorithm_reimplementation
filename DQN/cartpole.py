# coding: utf-8
# 达到了solving标准，开心！https://gym.openai.com/envs/CartPole-v0/ “CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.“
import gym
import math
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import count

# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np_default_dtype = np.float32
torch_default_dtype = torch.float32
torch.set_default_dtype(torch_default_dtype)

min_probability = 0.05
max_probability = 0.95
steps_threshold = 400
gamma = 0.99
target_update = 10
learning_rate = 5 * 1e-3

n_episodes = 100
batch_size = 32

# Num	Observation	Min	Max
# 0	Cart Position	-2.4	2.4
# 1	Cart Velocity	-Inf	Inf
# 2	Pole Angle	~ -41.8°	~ 41.8°
# 3	Pole Velocity At Tip	-Inf	Inf
# Cart Position，负数为左边，正数为右边
# Pole Angle，负数为左倾斜，正数为右倾斜
# action，0为向左移动，1为向右移动

# 这个函数需要重写，但框架放这，免得自己写了以后要调整的太多了
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = pd.DataFrame(columns = ['current_state', 'action', 'next_state', 'reward'])
        
    def push(self, row_dict):
        self.memory = self.memory.append([row_dict]).reset_index(drop=True)
        if len(self.memory) > self.capacity:
            self.memory = self.memory.iloc[-self.capacity:, :].reset_index(drop = True)

    def sample(self, batch_size = batch_size):
        return self.memory.sample(batch_size).reset_index(drop = True)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

def select_action(policy_net, current_state, steps_done):
    # current_state = current_state.unsqueeze(0)
    random_action_probability = min_probability + (max_probability - min_probability) * 1 / (1+math.exp(steps_done/steps_threshold)) # 随着steps_done增大，random_action_probability逐渐从0.95降至0.05
    if np.random.rand() < random_action_probability: # 随机选
        return random.sample([0,1], 1)[0]
    else: 
        with torch.no_grad():
            return policy_net(current_state).max(1)[1].item()

def select_action_after_training(policy_net, current_state):
    with torch.no_grad():
        # print(policy_net(current_state), type(policy_net(current_state)), policy_net(current_state).shape, 'policy_net(current_state)')
        return policy_net(current_state).max(1)[1].item()

def optimize_policy_net(policy_net, target_net, optimizer, memory_sample):
    mask = memory_sample['next_state'].notnull().tolist()
    nonfinal_memory_sample = memory_sample[memory_sample['next_state'].notnull()]
    current_state_batch = torch.cat(memory_sample.loc[:, 'current_state'].tolist(), 0)
    next_state_batch = torch.cat(nonfinal_memory_sample.loc[:, 'next_state'].tolist(), 0)
    action_batch = torch.tensor(memory_sample['action'].tolist(), dtype = torch.int64).unsqueeze(1)
    reward_batch = torch.tensor(memory_sample['reward'].tolist())

    # print(reward_batch, reward_batch.shape, 'reward_batch')
    # print(action_batch, action_batch.shape, 'action_batch')

    current_q = policy_net(current_state_batch).gather(1, action_batch).squeeze()
    next_q = torch.zeros(batch_size, dtype=torch_default_dtype)
    next_q[mask] = target_net(next_state_batch).max(1)[0].detach()
    # print(memory_sample['next_state'].notnull(), type(memory_sample['next_state'].notnull()), "memory_sample['next_state'].notnull()")
    # print(next_q, type(next_q), next_q.shape, 'next_q')
    expected_current_q = reward_batch + gamma*next_q

    optimizer.zero_grad()
    # Huber implementation:  F.smooth_l1_loss
    loss = F.mse_loss(current_q, expected_current_q)
    loss.backward()
    optimizer.step()
    # print(f"loss: {loss.item()}")
    return policy_net, target_net, optimizer

def evaluate(env, policy_net):
    # CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials. 参考https://gym.openai.com/envs/CartPole-v0/
    n_games = 100
    wining_threshold = 195
    total_reward = 0
    for i in range(n_games):
        current_state = env.reset()
        # env.render()
        current_state = torch.from_numpy(current_state.astype(np_default_dtype)).view(1, -1)
        episode_reward = 0
        # self.env.render(observation)
        for j in count():
            action = select_action_after_training(policy_net, current_state)
            # action = select_action(policy_net, current_state, 500)
            current_state, reward, done, info = env.step(action)
            # env.render()
            current_state = torch.from_numpy(current_state.astype(np_default_dtype)).view(1, -1)
            episode_reward += reward
            if done:
                # print(f"game ends after {j+1} action")
                break
            if episode_reward >= wining_threshold * 3:
                break
        total_reward += episode_reward
        # print(f"game {i+1} ends")
        # print('-'*30)
    print(f"avg reward >= {total_reward/n_games}")

def play(env, policy_net):
    try:
        current_state = env.reset()
        env.render()
        current_state = torch.from_numpy(current_state.astype(np_default_dtype)).view(1, -1)
        env.render()
        for j in count():
            action = select_action_after_training(policy_net, current_state)
            # action = select_action(policy_net, current_state, 500)
            current_state, reward, done, info = env.step(action)
            env.render()
            current_state = torch.from_numpy(current_state.astype(np_default_dtype)).view(1, -1)
            if done:
                print(f"game ends after {j+1} action")
                break
    except KeyboardInterrupt:
        print(f"game ends after {j+1} action")


def evaluate_random(env):
    # CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials. 参考https://gym.openai.com/envs/CartPole-v0/
    n_games = 10
    # wining_threshold = 195
    total_reward = 0
    for i in range(n_games):
        current_state = env.reset()
        current_state = torch.from_numpy(current_state.astype(np_default_dtype)).view(1, -1)
        episode_reward = 0
        # self.env.render(observation)
        for j in count():
            action = random.sample([0,1], 1)[0]
            current_state, reward, done, info = env.step(action)
            current_state = torch.from_numpy(current_state.astype(np_default_dtype)).view(1, -1)
            episode_reward += reward
            if done:
                # print(f"game ends after {j+1} action")
                break
        total_reward += episode_reward
        print(f"game {i+1} ends")
    print(f"avg reward is {total_reward/n_games}")

def evaluate_simple(env):
    def select_action_simple(current_state):
        if current_state[3] < 0:
            return 0
        elif current_state[3] > 0:
            return 1
        elif current_state[3] == 0:
            if current_state[2] < 0:
                return 1
            elif current_state[2] >=0:
                return 0
    # CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials. 参考https://gym.openai.com/envs/CartPole-v0/
    n_games = 10
    # wining_threshold = 195
    total_reward = 0
    for i in range(n_games):
        current_state = env.reset()
        # current_state = torch.from_numpy(current_state.astype(np_default_dtype)).view(1, -1)
        env.render()
        episode_reward = 0
        # self.env.render(observation)
        for j in count():
            action = select_action_simple(current_state)
            current_state, reward, done, info = env.step(action)
            env.render()
            # current_state = torch.from_numpy(current_state.astype(np_default_dtype)).view(1, -1)
            episode_reward += reward
            if done:
                # print(f"game ends after {j+1} action")
                break
        total_reward += episode_reward
        print(f"game {i+1} ends")
    print(f"avg reward is {total_reward/n_games}")

def evaluate_move0(env):
    # CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials. 参考https://gym.openai.com/envs/CartPole-v0/
    n_games = 100
    # wining_threshold = 195
    total_reward = 0
    for i in range(n_games):
        current_state = env.reset()
        # current_state = torch.from_numpy(current_state.astype(np_default_dtype)).view(1, -1)
        # env.render()
        episode_reward = 0
        # self.env.render(observation)
        for j in count():
            action = 0
            current_state, reward, done, info = env.step(action)
            # env.render()
            # current_state = torch.from_numpy(current_state.astype(np_default_dtype)).view(1, -1)
            episode_reward += reward
            if done:
                # print(f"game ends after {j+1} action")
                break
        total_reward += episode_reward
        print(f"game {i+1} ends")
    print(f"avg reward is {total_reward/n_games}")

def train(env):
    # 注意，每次reward为1，那么训练的动力在于done之后的q为0，其它时候的q为>0，那么每次done之后q为0一定要添加近memory里，否则训练不出来，这bug找了好久，我醉了要

    steps_done = 0
    memory = ReplayMemory(10000)
    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    # optimizer = optim.RMSprop(policy_net.parameters(), lr = learning_rate)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    for i_episode in range(1, n_episodes+1):
        current_state = env.reset()
        # env.render()
        current_state = torch.from_numpy(current_state.astype(np_default_dtype)).view(1, -1)
        # print(current_state, current_state.dtype, type(current_state), current_state.shape, 'current_state')
        for j in count():
            action = select_action(policy_net, current_state, steps_done)
            next_state, reward, done, info = env.step(action)
            # reward = -abs(next_state[3])*10
            # print(next_state, next_state.shape, 'next_state')
            # print(reward, type(reward), 'reward')
            # env.render()
            next_state = torch.from_numpy(next_state.astype(np_default_dtype)).view(1, -1)
            steps_done += 1

            row_dict = {
                'current_state': current_state,
                'action': action,
                'next_state': next_state,
                'reward': reward
            }
            memory.push(row_dict)
            current_state = next_state
            if len(memory) >= batch_size:
                policy_net, target_net, optimizer = optimize_policy_net(policy_net, target_net, optimizer, memory.sample())
            if done:
                row_dict = {
                    'current_state': current_state,
                    'action': action,
                    'next_state': None,
                    'reward': reward
                }
                memory.push(row_dict)
                print(f"episode ends after {j+1} steps")
                break
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print(f"episode {i_episode} done")
    return policy_net, target_net

def save_model(policy_net, file = 'p.pt'):
    torch.save(policy_net, file)

def load_model(file = 'p.pt'):
    return torch.load(file)

def main():
    env = gym.make('CartPole-v0').unwrapped
    policy_net, _ = train(env)
    save_model(policy_net)

    policy_net = load_model('p.pt')
    evaluate(env, policy_net)
    # play(env, policy_net)

            
            # print(current_state, current_state.shape, type(current_state), 'current_state')
    pass

def test():
    # for i in range(10):
    #     a = random.sample([0,1],1)[0]
    #     print(a)
    env = gym.make('CartPole-v1').unwrapped
    evaluate_move0(env)
    pass

if __name__ == '__main__':
    main()
    # test()

    pass




