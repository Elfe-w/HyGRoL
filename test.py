import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context="talk")
# %matplotlib inline


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.view(1,x.shape[0],-1)
        y = self.pe[:, :x.size(1)]
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        x = x.view(x.shape[1],-1)
        return self.dropout(x)


plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(th.zeros(100, 20)))
print(y.shape)