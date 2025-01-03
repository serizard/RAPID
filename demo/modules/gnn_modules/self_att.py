import os
import pandas as pd
import numpy as np
from pprint import pprint
# torch:
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm            


class Attention(nn.Module):
    def __init__(self, device,hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.device = device
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(torch.tanh(inputs),
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze(-1)), dim=-1) #F.relu # , dim=-1

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).to(self.device)


        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

        attentions = masked.div(_sums)

        if attentions.dim() == 1:
            attentions = attentions.unsqueeze(1)
            
        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        #representations = weighted.sum(1).squeeze()

        return weighted, attentions