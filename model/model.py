from gcnlayer import GCNConv
from gatlayer import GATConv
import torch as th
import dgl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, copy, time
from torch.autograd import Variable

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

class Gating(nn.Module):
    def __init__(self,heads,in_feature,device):
        super(Gating,self).__init__()
        self.heads=heads
        self.lq=nn.Linear(in_feature, in_feature // heads)
        self.lk1 = nn.Linear(in_feature, in_feature // heads)
        self.lk2 = nn.Linear(in_feature, in_feature // heads)
        self.lv1 = nn.Linear(in_feature, in_feature // heads)
        self.lv2 = nn.Linear(in_feature, in_feature // heads)
        self.lh = nn.Linear( in_feature, in_feature)
        self.scale = th.sqrt(th.FloatTensor([in_feature // heads])).to(device)
    def forward(self,Q,K,V):
        list_concat = []
        for i in range(self.heads):
            q=self.lq(Q)
            k1=self.lk1(K)
            k2=self.lk2(V)
            v1=self.lv1(K)
            v2=self.lv2(V)

            kv1=th.sum(th.mul(q,k1), dim=(1,), keepdim=True)
            kv2 = th.sum(th.mul(q, k2), dim=(1,), keepdim=True)

            kv1_1 = kv1 - th.max(kv1, kv2)
            kv2_1 = kv2 - th.max(kv1, kv2)

            kv1 = th.exp(kv1_1)
            kv2= th.exp(kv2_1)
            kv1_S = kv1 / (kv1 + kv2)
            kv2_S = kv2 / (kv1 + kv2)


            kv1_S =th.mul(kv1_S ,v1)
            kv2_S =th.mul(kv2_S ,v2)
            list_concat.append(kv1_S+ kv2_S)  # self.headattention_qkv(W_q, W_k, W_v, mask, flag, antimask))

        concat_head = th.cat(list_concat, -1)
        W_o = self.lh(concat_head)
        return W_o
def pooling(node_num,node_emb):
    import copy
    node_begin=0
    node=[]
    for i in range(len(node_num)):
        node_slice = node_emb[node_begin:node_begin + node_num[i]]
        node_begin=node_begin+node_num[i]
        node.append(th.sum(node_slice,0))
    ast_node= th.stack(node)
    return ast_node
class JKNet(th.nn.Module):
    """An implementation of Jumping Knowledge Network (arxiv 1806.03536) which
    combine layers with concatenation.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        n_layers (int): Number of the convolution layers.
        n_units (int): Size of the middle layers.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """
    def __init__(self, in_features, gat_head =2,gating_head=2,n_layers=2, n_units=128,dropout=0.5,allow_zero_in_degree=True,device='cpu'):
        super(JKNet, self).__init__()
        self.n_layers = n_layers
        self.gcnconv0 = GCNConv(in_features, n_units,allow_zero_in_degree= allow_zero_in_degree)
        self.gatconv0 = GATConv(in_features, n_units,gat_head,allow_zero_in_degree= allow_zero_in_degree)
        self.dropout0 = th.nn.Dropout(dropout)
        self.gating = Gating(gating_head, n_units, device)
        self.pooling = pooling
        self.pe = PositionalEncoding(in_features,dropout)
        for i in range(1, self.n_layers):
            setattr(self, 'gcnconv{}'.format(i),
                    GCNConv(n_units, n_units,allow_zero_in_degree= allow_zero_in_degree))
            setattr(self, 'gatconv{}'.format(i),
                    GATConv(n_units, n_units,gat_head,allow_zero_in_degree= allow_zero_in_degree))
            setattr(self, 'dropout{}'.format(i), th.nn.Dropout(0.5))
        self.concatLayer1 = nn.Linear(n_layers * n_units, n_units)
        self.concatLayer2 = nn.Linear(n_layers * n_units, n_units)
    def forward(self, node_num,sg,edfg,Feat,sgFeat,edfgFeat):
        layer_outputs1 = []
        layer_outputs2 = []
        Feat1 = self.pe.forward(Feat)
        Feat1 = self.gcnconv0(sg,Feat1,sgFeat)
        Feat2 = self.gatconv0(edfg,Feat,edfgFeat)
        for i in range(self.n_layers):
            dropout = getattr(self, 'dropout{}'.format(i))
            gcnconv = getattr(self, 'gcnconv{}'.format(i))
            gatconv = getattr(self, 'gatconv{}'.format(i))
            out1 = dropout(F.relu(gcnconv(sg,Feat1,sgFeat)))
            out2 = dropout(F.relu(gatconv(edfg,Feat2,edfgFeat)))
            layer_outputs1.append(out1)
            layer_outputs2.append(out2)

        h1 = th.cat(layer_outputs1, dim=1)
        h2 = th.cat(layer_outputs2, dim=1)
        h1 = self.concatLayer1(h1)
        h2 = self.concatLayer2(h2)
        node_feat = self.gating(h1, h1, h2)
        ast_feat = self.pooling(node_num, node_feat)
        #最后再进行分类
        # print('h1',h1.shape)
        # print('h2',h2.shape)
        return ast_feat


class CLA(th.nn.Module):
    """An implementation of Jumping Knowledge Network (arxiv 1806.03536) which
    combine layers with concatenation.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        n_layers (int): Number of the convolution layers.
        n_units (int): Size of the middle layers.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """

    def __init__(self, in_features, out_features, gat_head=2, gating_head=2, n_layers=4, n_units=10, dropout=0.5,
                 allow_zero_in_degree=True, device='cpu'):
        super(CLA, self).__init__()
        self.jknet = JKNet(in_features, gat_head=gat_head,
                gating_head=gating_head, n_layers=n_layers,
                n_units=n_units, dropout=dropout,
                 allow_zero_in_degree=allow_zero_in_degree, device=device)
        self.cla = nn.Sequential(
            nn.Linear( n_units, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_features),
        )
    def forward(self, node_num, sg, edfg, Feat, sgFeat, edfgFeat,pre_model_path = ""):
        if pre_model_path:
            self.jknet.load_state_dict(torch.load(pre_model_path))
        ast_feat = self.jknet(node_num, sg, edfg, Feat, sgFeat, edfgFeat)
        res = self.cla(ast_feat)
        return res

class CLO(th.nn.Module):
    """An implementation of Jumping Knowledge Network (arxiv 1806.03536) which
    combine layers with concatenation.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        n_layers (int): Number of the convolution layers.
        n_units (int): Size of the middle layers.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """

    def __init__(self, in_features, out_features, gat_head=2, gating_head=2, n_layers=4, n_units=128, dropout=0.5,
                 allow_zero_in_degree=True, device='cpu'):
        super(CLO, self).__init__()
        self.jknet = JKNet(in_features, gat_head=gat_head,
                gating_head=gating_head, n_layers=n_layers,
                n_units=n_units, dropout=dropout,
                 allow_zero_in_degree=allow_zero_in_degree, device=device)
        self.cla = nn.Sequential(
            nn.Linear( n_units, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_features),
        )
    def forward(self, node_num1, sg1, edfg1, Feat1, sgFeat1, edfgFeat1,
                node_num2, sg2, edfg2, Feat2, sgFeat2, edfgFeat2,pre_model_path = ""):
        if pre_model_path:
            self.jknet.load_state_dict(torch.load(pre_model_path))
        ast_out1 = self.jknet(node_num1, sg1, edfg1, Feat1, sgFeat1, edfgFeat1)
        ast_out2 = self.jknet(node_num2, sg2, edfg2, Feat2, sgFeat2, edfgFeat2)

        ast_out = th.abs(th.add(ast_out1, -ast_out2))
        res = self.cla(ast_out)
        res = th.sigmoid(res)
        return res


if __name__=="__main__":
    import networkx as nx
    import matplotlib.pyplot as plt

    g1 = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3]))
    g2 = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]))
    node_num = [2,1,3]
    # g2 = dgl.DGLGraph()
    # g2.add_nodes(10)
    # edfgsrc=[0,1]
    # edfgdst=[8,9]
    # g2.add_edges(edfgsrc,edfgdst)
    # nx.draw(g2.to_networkx(), with_labels=True)  # 可视化图
    # plt.show()
    # g = dgl.add_self_loop(g)

    feat = th.ones(6, 10)
    sgfeat = th.ones(6,10)
    edfgfeat = th.ones(5, 10)
    conv = CLO(10, 2,n_units=10)
    res = conv(node_num,g1,g2, feat,sgfeat, edfgfeat,node_num,g1,g2, feat,sgfeat, edfgfeat)
    print(res.shape)