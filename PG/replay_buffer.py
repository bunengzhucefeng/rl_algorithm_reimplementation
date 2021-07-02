import gym
import numpy as np
from numpy.core.numeric import indices
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size


        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        if self.obs is None:
            return 0
        else:
            return self.obs.shape[0]

    def add_rollouts(self, rollout):
        # rollout是一个字典，它的key包括'ob','ac','rew','next_ob','terminal'
        if self.obs is None:
            self.obs = np.array([rollout['ob']])
            self.acs = np.array([rollout['ac']])
            self.rews = np.array([rollout['rew']])
            self.next_obs = np.array([rollout['next_ob']])
            self.terminals = np.array([rollout['terminal']])
        else:
            self.obs = np.concatenate((self.obs, np.expand_dims(rollout['ob'],0)))
            self.acs = np.concatenate((self.acs, np.expand_dims(rollout['ac'],0)))
            self.rews = np.concatenate((self.rews, np.expand_dims(rollout['rew'],0)))
            self.next_obs = np.concatenate((self.next_obs, np.expand_dims(rollout['next_ob'],0)))
            self.terminals = np.concatenate((self.terminals, np.expand_dims(rollout['terminal'],0)))

    def sample_random_data(self, batch_size):
        assert (
                self.obs.shape[0]
                == self.acs.shape[0]
                == self.rews.shape[0]
                == self.next_obs.shape[0]
                == self.terminals.shape[0]
        )

        ## TODO return batch_size number of random entries from each of the 5 component arrays above
        ## HINT 1: use np.random.permutation to sample random indices
        ## HINT 2: return corresponding data points from each array (i.e., not different indices from each array)
        ## HINT 3: look at the sample_recent_data function below
        indices = np.random.permutation(len(self))[0:batch_size]
        # return TODO, TODO, TODO, TODO, TODO
        return self.obs[indices], self.acs[indices], self.rews[indices], self.next_obs[indices], self.terminals[indices]

    def sample_recent_data(self, batch_size=1):
        return (
            self.obs[-batch_size:],
            self.acs[-batch_size:],
            self.rews[-batch_size:],
            self.next_obs[-batch_size:],
            self.terminals[-batch_size:],
        )