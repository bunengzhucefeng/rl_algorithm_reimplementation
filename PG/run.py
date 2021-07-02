import torch
import numpy as np
import gym

import global_var as gv
import pytorch_util as ptu


from mlp_policy import MLP_Policy
from pg_agent import PG_Agent

'''
TODO:
处理r随着t的变化而变化
分布式收集数据
处理带cuda的形式
'''

torch.set_default_dtype(gv.torch_default_type)
np_default_type = gv.np_default_type

def test():

    a = np.arange(4).reshape(2,2)
    b = np.expand_dims(np.array([100, 103]), axis = 0)
    c = [a,b]
    d = np.concatenate(c)
    print(d)
    return

    # 测试nn.Sequential在固定模式下，可以支持的输入的shape有哪些
    policy = ptu.build_mlp(input_size=10,
                                output_size=5,
                                n_layers=3,
                                size = 8
                                )
    # print(policy.state_dict())
    a = torch.from_numpy(np.random.rand(20,6,10).astype(np_default_type))
    print(type(policy), 'type policy')
    print(type(policy)==torch.nn.modules.container.Sequential)
    print('-'*20)

    print(a)
    print(a.shape)
    print('-'*20)

    b = policy(a)
    print(b)
    print(b.shape)
    pass



def main():


    save_path = './cartpole.pth'

    env = gym.make('CartPole-v0')

    ac_dim = 2
    ob_dim = 4
    n_layers = 2
    size = 4

    policy = ptu.build_mlp(
        ob_dim,
        ac_dim,
        n_layers,
        size
    )
    # print(policy.parameters(), type(policy.parameters()))
    # return
    learning_rate = 1e-2
    n_policy = 100 # 迭代多少个策略
    n_episode = 100 # 每个策略下输出多少个episode用来更新该策略
    max_timesteps = 500 # 最多一个episode多个步，免得一个很强的策略出来以后episode不终止了
    agent = PG_Agent(
        env,
        policy,
        learning_rate,
        n_policy,
        n_episode,
        max_timesteps
    )
    # agent.load_policy()
    # agent.generate_episode(render=True)
    # return

    print('start training')
    agent.train()
    print('end training')
    # agent.save_policy()
    # agent.policy.save(save_path)

    pass


if __name__ == '__main__':
    # test()
    main()

