# 总管buffer和policy



from os import path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
from torch.serialization import load

import global_var as gv

torch.set_default_dtype(gv.torch_default_type)

class PG_Agent(object):
    def __init__(
        self,
        env,
        policy: torch.nn.modules.container.Sequential, 
        learning_rate: float,
        n_policy: int, # 迭代多少个策略
        n_episode: int, # 每个策略下输出多少个episode用来更新该策略
        max_timesteps: int # 最多一个episode多个步，免得一个很强的策略出来以后episode不终止了
    ) -> None:
        super().__init__()
        self.env = env
        self.policy = policy
        self.learning_rate = learning_rate
        # self.buffer = buffer
        self.n_policy = n_policy
        self.n_episode = n_episode
        self.max_timesteps = max_timesteps

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def get_acs(self, obs):
        '''
        obs is shape (batch_size, n_dim)
        '''
        logits = self.policy(obs)
        acs = torch.argmax(logits, dim=1)
        return acs # shape (batch_size,)
    
    def get_ac(self, ob):
        '''
        ob is shape (n_dim,)
        '''
        if isinstance(ob, np.ndarray):
            ob = torch.from_numpy(ob.astype(gv.np_default_type))
        logits = self.policy(ob.view(1,-1))
        # 按照概率分布来获取ac，而不是直接取较大Logit者，这里dubug了好久，烦烦烦
        # ac = torch.argmax(logits)
        distri = distributions.Categorical(logits=logits)

        return distri.sample().item()

    def generate_episode(self, render = False):
        next_ob = self.env.reset().reshape(1,-1)
        if render:
            self.env.render()
        timesteps = 0
        obs = []
        acs = []
        next_obs = []
        res = []
        terminals = []
        while True:
            ob = next_ob
            ac = self.get_ac(ob)
            next_ob, re, done, info = self.env.step(ac)
            if render:
                self.env.render()
            next_ob = next_ob.reshape(1,-1)
            obs.append(ob)
            acs.append(ac)
            next_obs.append(next_ob)
            res.append(re)
            terminals.append(done)
            # break
            if done or timesteps > self.max_timesteps:
                break
        # print(acs, type(acs), 'acs')
        return torch.from_numpy(np.concatenate(obs).astype(gv.np_default_type)), torch.tensor(acs), torch.from_numpy(np.concatenate(next_obs)), torch.tensor(res), torch.tensor(terminals)


    def train(self):
        '''
        for _ in 轮数：
            由于不知道如何处理不同的episode的timesteps不一样的问题，所以设置batch_size为1，每次只处理一个episode
            # 那么也不需要buffer了
                按照既有策略生成buffer
                从buffer中获取数据

            利用loss计算j tilder
            求梯度
            更新loss

        '''
        # print(self.policy.state_dict(), 'p1')
        for i_policy in range(self.n_policy):
            J = 0 # j tilda，也就是loss
            q = 0
            for i_episode in range(self.n_episode):
                # 生成
                obs, acs, next_obs, res, terminals = self.generate_episode()
                # print(acs, acs.shape, 'acs')
                assert(len(obs)==len(next_obs)==len(res)==len(acs)==len(terminals))
                r_tau = sum(res)
                logits = self.policy(obs)

                # print(logits, logits.shape, 'logits')
                # print(acs, type(acs))

                criterion = nn.CrossEntropyLoss(reduction='sum') # 注意这里要选择sum才对，否则和policy gradient的公式并不一样，导致训练一直没有效果，难受啊，找了好久这个问题
                negative_likelihoods = criterion(logits, acs)
                # print(negative_likelihoods, negative_likelihoods.shape, 'negative_likelihoods')
                negative_likelihoods = negative_likelihoods.sum()
                # print(negative_likelihoods, negative_likelihoods.shape, 'negative_likelihoods')
                # print(r_tau, 'r_tau')
                J += negative_likelihoods*r_tau
                q += res.sum().item()
                
            J /= self.n_episode
            self.optimizer.zero_grad()
            print(f"第{i_policy}个策略的loss J tilda 为 {J.item()}, avg return >= {q/self.n_episode}") # 这里的loss估计不对，要用平均每次的
            J.backward()
            self.optimizer.step()

            # print(self.policy.state_dict(), 'p2')

    def save_policy(self, path='policy.pth'):
        torch.save(self.policy, path)

    def load_policy(self, path='policy.pth'):
        self.policy = torch.load(path)


