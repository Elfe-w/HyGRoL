import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
import pandas as pd
import numpy as np
from numpy import array
#import tensorflow as tf
import re
file = ["AST.txt", "ADG.txt"]
embedding = TransformerDocumentEmbeddings('bert-base-uncased')
edgDict = [{}, {}]
edgList = [[], []]
src = [[], []]
dst = [[], []]

def getEmbedding(op):
    df = pd.read_table(file[op])
    embeddingList = []
    nodeNum = 0
    begin = 0
    if op == 0:
        begin = 1
    for i in range(begin, len(df)):
        s = df.values[i]
        if s[0].find('->') == -1:
            n = s[0].find('[')
            if op == 1:
                n = s[0].find(':')

            sentence = Sentence(s[0][n:len(s[0])])
            embedding.embed(sentence)
            list = sentence.embedding.detach().cpu().numpy().tolist()
            print(list)
            embeddingList.append(list)
            nodeNum = nodeNum + 1
        else:
            print('embedding shpae:{}\n'.format(np.array(embeddingList).shape))
            # print(embeddingList)
            return embeddingList, nodeNum

def edgInfo(op):
    pre_vex = []
    next_vex = []
    vexNum = 0
    with open(file[op], "r") as f:
        ID = 0
        data = f.readline()
        while data:
            if data.find("->") == -1:
                matchObj = re.match(r'(\"*)(\d+)(\"*)', data)
                if matchObj:
                    edgDict[op][matchObj.group()] = ID
                    ID += 1
                    vexNum += 1
            else:
                if op == 0:
                    sData = data.strip().split('->')
                    edgList[op].append(sData)
                else:
                    edgeInfo = data.strip().split(':')[0]
                    sData = edgeInfo.split('->')
                    edgList[op].append(sData)
            data = f.readline()

    adjMatrix= [([0] * vexNum) for i in range(vexNum)]
    for i in range(len(edgList[op])):
         pre = edgDict[op].get(edgList[op][i][0].strip())
         next = edgDict[op].get(edgList[op][i][1].strip())
         pre_vex.append(pre)
         next_vex.append(next)
    src[op] = np.array(pre_vex)
    dst[op] = np.array(next_vex)

def build_karate_club_graph(op):
    # All 78 edges are stored in two numpy arrays. One for source endpoints
    # while the other for destination endpoints.
    edgInfo(op)
    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src[op], dst[op]])
    v = np.concatenate([dst[op], src[op]])
    # Construct a DGLGraph
    return dgl.graph((u, v))
# G=build_karate_club_graph()

def visual(G):
    # 可视化
    nx_G = G.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(nx_G) ## 生成节点位置
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    # plt.pause(10)


from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        res1 = self.conv1(g, inputs)
        res2 = torch.relu(res1)
        res1 = self.conv2(g, res2)
        return res1, res2

def train(G, inputs, labeled_nodes, labels, nodeNum):
    net = GCN(768, 128, nodeNum)

    import itertools
    print(inputs.dtype)
    optimizer = torch.optim.Adam(itertools.chain(net.parameters()), lr = 0.01)
    all_logits = []
    for epoch in range(50):
        logits, res2 = net(G, inputs)
        # we save the logits for visualization later
        all_logits.append(logits.detach()) # detach代表从当前计算图中分离下来的
        logp = F.log_softmax(logits, 1)

        loss = F.nll_loss(logp[labeled_nodes], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
        if(epoch == 49):
            print('GCN res:{}\n'.format(logits.shape))

def fusion(embed0, embed1):
    ans = embed1
    for old_adg in edgDict[1]:
        # print("--------------------------", old_adg)
        old_ast = -1
        flag = 0
        for old_ast in edgDict[0]:
            # print(old_ast, edgDict[0][old_ast])
            if (int)(edgDict[0][old_ast]) == (int)(old_adg):
                flag = 1
                break
        # print(old_ast)
        if flag == 0:
            continue
        new_ast = edgDict[0][old_ast]
        new_adg = edgDict[1][old_adg]
        # print("! ", old_adg, new_adg)
        # print("! ", old_ast, new_ast)
        for i in range(len(ans[new_adg])):
            ans[new_adg][i] += embed0[new_ast][i]
    return ans


def main():
    G_ast = build_karate_club_graph(0)
    G_adg = build_karate_club_graph(1)
    G = [G_ast, G_adg]

    print("ast中节点数:", G[0].number_of_nodes(), " ast中边数:", G[0].number_of_edges())
    print("adg中节点数:", G[1].number_of_nodes(), " adg中边数:", G[1].number_of_edges())

    visual(G_ast), visual(G_adg)
# # -----------------------------------------------------
#     print(edgDict[0])
#     print(edgList[0])
#     print("_____________")
#     print(edgDict[1])
#     print(edgList[1])
# # -----------------------------------------------------

    embed0, num0 = getEmbedding(0)
    embed1, num1 = getEmbedding(1)
    embed0 = torch.from_numpy(np.array(embed0))
    embed1 = torch.from_numpy(np.array(embed1))
    train(G[0], embed0.float(), torch.tensor([0, num0 - 1]), torch.tensor([0, num0 - 1]),128)
    train(G[1], embed1.float(), torch.tensor([0, num1 - 1]), torch.tensor([0, num1 - 1]),128)

    embed = fusion(embed0, embed1)
    print(embed)

main()