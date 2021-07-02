# coding: utf-8

import torch
import gym

import global_var as gv

from ppo_agent import PPO_Agent
from pytorch_util import build_mlp

torch.set_default_dtype(gv.torch_default_dtype)
'''
TODO:
- 多进程generate episode
- 批量计算Lclip，提高效率一下
'''
def gen_parameters():
    env = gym.make('CartPole-v0')

    ac_dim = 2
    ob_dim = 4
    n_layers = 2
    size = 4
    policy = build_mlp(
        ob_dim,
        ac_dim,
        n_layers,
        size
    )

    learning_rate = 1e-2
    n_policy = 30
    n_episode = 25
    epsilon = 0.2
    max_timesteps = 500

    return (
        env, 
        policy,
        learning_rate,
        n_policy,
        n_episode,
        epsilon,
        max_timesteps
    )

def train():
    agent = PPO_Agent(
        *gen_parameters()
    )

    agent.train()


    save_path = './policy.pth'
    agent.save_policy(save_path)


def main():
    train()
    # check_saved_policy()


    pass

def check_saved_policy():
    save_path = './policy.pth'
    agent = PPO_Agent(
        *gen_parameters()
    )
    agent.load_policy(save_path)
    for i in range(10):
        obs = agent.generate_episode(render=True)[0]
        print(f"episode length is {obs.shape[0]}")

def test():
    import numpy as np
    t = torch.from_numpy(np.arange(10).astype(gv.np_default_dtype)).view((5,2))
    print(t, t.shape, 't')
    import torch.nn as nn
    m = nn.Softmax(dim=1)
    t_softmax = m(t)
    print(t_softmax, t_softmax.shape, 't_softmax')
    pass


if __name__ == '__main__':
    main()
    # test()