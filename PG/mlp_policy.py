import torch
import torch.nn as nn

import pytorch_util as ptu
import replay_buffer

class MLP_Policy(nn.Module):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 ):
        super().__init__()
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.policy = ptu.build_mlp(input_size=self.ob_dim,
                                    output_size=self.ac_dim,
                                    n_layers=self.n_layers,
                                    size = self.size
                                    )

    def save(self, file_path):
        torch.save(self.policy.state_dict(), file_path)
    
    def load(self, file_path):
        self.policy.load_state_dict(torch.load(file_path))

        




