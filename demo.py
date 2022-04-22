from dataset import loadJava,loadOJ,javaParser
# from model import model,losses
from gensim.models.word2vec import Word2Vec, LineSentence
from tqdm import tqdm
import dataset
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from dataset.cParser import cloPipline,claPipline
from model.model import CLO,CLA
import dgl
import numpy as np
import torch as th
Dataset = loadOJ.OJ_DATASET(file_path='data/oj_funcs.pkl', clone_path='data/oj_pair_ids.pkl', clone=False)
dataloader = DataLoader(Dataset, batch_size=2, shuffle=True)
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

def node2emb(nodeInfo,keyword = 'oj'):
    nodeemb = []
    embSavePath = 'data/emb/'+keyword+'_w2v_128'
    word2vec = Word2Vec.load(embSavePath).wv
    vocab = word2vec.vocab
    if type(nodeInfo).__name__ == 'dict':
        for k,v in nodeInfo.items():
            str = v.replace("\n","")
            list = str.split()
            nodeemb.append(word2vec[list[0]])
    else:
        if type(nodeInfo).__name__ == 'list':
            for v in nodeInfo:
                str = v.replace("\n", "")
                list = str.split()
                nodeemb.append(word2vec[list[0]])
    return nodeemb
model = CLA(128,2,n_units=128).to(device)
for i, item in enumerate(tqdm(dataloader)):
    index, code, label = item
    nodeInfo, srcSg, srcEdfg, dstSg, dstEdfg, nodeNum, sgInfo, edfgInfo = claPipline(code)
    print('nodeNum1',len(nodeInfo),nodeNum)
    Feat = th.FloatTensor(node2emb(nodeInfo)).to(device)
    sg = dgl.graph((srcSg, dstSg)).to(device)
    edfg = dgl.graph((srcEdfg,dstEdfg)).to(device)
    sgFeat = th.FloatTensor(node2emb(sgInfo)).to(device)
    edfgFeat = th.FloatTensor(node2emb(edfgInfo)).to(device)
    # print('sg',len(srcSg),len(dstSg),len(srcEdfg),len(dstEdfg),len(sgInfo),len(edfgInfo))

    res = model(nodeNum, sg, edfg, Feat, sgFeat, edfgFeat)
    print(res)
    # # print(batchAstNode)
    # break

# print(wrongData)