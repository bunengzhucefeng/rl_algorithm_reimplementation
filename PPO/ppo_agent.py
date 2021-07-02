# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.optim as optim
import copy


import global_var as gv

torch.autograd.set_detect_anomaly(True)

class PPO_Agent(nn.Module):
    def __init__(self, env, policy, learning_rate, n_policy, n_episode, epsilon, max_timesteps):
        '''
        input: policy, feedforward neural network
               env, cart-pole的gym env
        '''
        super().__init__()
        self.env = env
        self.policy = policy
        self.previous_policy = copy.deepcopy(policy) # 注意这里一定要deepcopy一下，否则会报RuntimeError: one of the variables needed for gradient computation has been modified
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.n_policy = n_policy
        self.n_episode = n_episode
        self.epsilon = epsilon
        self.max_timesteps = max_timesteps

    def get_ac(self, ob):
        '''
        ob is shape (ob_dim,)
        '''
        if isinstance(ob, np.ndarray):
            ob = torch.from_numpy(ob.astype(gv.np_default_dtype))
        logits = self.policy(ob.view(1,-1))
        # 按照概率分布来获取ac，而不是直接取较大Logit者，这里dubug了好久，烦烦烦
        # ac = torch.argmax(logits)
        distri = distributions.Categorical(logits=logits)

        return distri.sample().item()

    # def get_probability(self, ob):
    #     '''
    #     ob is shape (ob_dim,)
    #     '''
    #     if isinstance(ob, np.ndarray):
    #         ob = torch.from_numpy(ob.astype(gv.np_default_dtype))
    #     logits = self.policy(ob.view(1,-1))
    #     m = nn.Softmax()
    #     probability = m(logits)
    #     return probability
    def get_probability(self, obs, acs, policy):
        '''
        obs shape (N, ob_dim)
        '''
        logits = policy(obs) # logits shape (N, ac_dim)
        assert(logits.shape == (len(obs), 2))
        m = nn.Softmax(dim=1)
        probabilities = m(logits) # probabilities shape (N, ac_dim)
        assert(probabilities.shape == (len(obs), 2))
        probabilities = probabilities.gather(1, acs)
        assert(probabilities.shape == (len(obs), 1))
        return probabilities


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
        return torch.from_numpy(np.concatenate(obs).astype(gv.np_default_dtype)), torch.tensor(acs).view(-1,1), torch.from_numpy(np.concatenate(next_obs).astype(gv.np_default_dtype)), torch.tensor(res).view((-1, 1)), torch.tensor(terminals)

    def calLclip(self, probability_ratio_t, advantage_t): # 计算ppo的代理函数Lclip
        surr1 = probability_ratio_t * advantage_t
        surr2 = torch.clamp(probability_ratio_t, 1-self.epsilon, 1+self.epsilon)*advantage_t
        return torch.min(surr1, surr2)
        if advantage_t >= 0:
            if probability_ratio_t >= 1+self.epsilon:
                return (1+self.epsilon)*advantage_t
            elif probability_ratio_t < 1+self.epsilon:
                return probability_ratio_t * advantage_t
        elif advantage_t < 0:
            if probability_ratio_t >= 1-self.epsilon:
                return probability_ratio_t * advantage_t
            elif probability_ratio_t < 1-self.epsilon:
                return (1-self.epsilon)*advantage_t

    @gv.func_info
    def train(self):
        for i_policy in range(1, self.n_policy+1):
            l = 0
            Lclip = 0
            for i_episode in range(1, self.n_episode+1):
                obs, acs, next_obs, res, terminals = self.generate_episode()
                # logits = self.policy(obs) # 
                Vt = self.policy(obs).gather(1, acs) # Vt表示V(St)
                Vt_plus1 = self.policy(next_obs).gather(1, acs) # Vt_plus1表示V(St+1)
                advantages = res + Vt - Vt_plus1
                previous_probabilities = self.get_probability(obs, acs, self.previous_policy).detach()
                probabilities = self.get_probability(obs, acs, self.policy)


                probability_ratios = probabilities / previous_probabilities  # 获取probability
                assert(probability_ratios.shape == (len(obs), 1))
                for t in range(0, obs.shape[0]):
                    probability_ratio_t = probability_ratios[t]
                    advantage_t = advantages[t:].sum()
                    Lclip = Lclip + self.calLclip(probability_ratio_t, advantage_t)
                l += acs.shape[0]
            
            self.previous_policy.load_state_dict(self.policy.state_dict())
            Lclip = Lclip / self.n_episode # 不要用/=，是inplace action了
            negative_Lclip = - Lclip
            self.optimizer.zero_grad()
            negative_Lclip.backward()
            self.optimizer.step()
            print(f"第{i_policy}次策略迭代，平均每个episode的代理函数值为{Lclip.item()}, 平均每个episode长度为{l/self.n_episode}")
        

    def save_policy(self, path):
        torch.save(self.policy, path)

    def load_policy(self, path):
        self.policy = torch.load(path)
        self.previous_policy.load_state_dict(self.policy.state_dict())


def test():
    


    t = torch.tensor([[1, 2], [3, 4]])

    print(t.shape)
    print(t.shape == (2,2))
    return

    # b = torch.gather(t, 1, torch.tensor([[0], [1]]))
    b = t.gather(1, torch.tensor([[0], [1]]))
    print(b, type(b), b.shape)

    pass

if __name__ == '__main__':
    test()