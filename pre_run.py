
import argparse
import dgl
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from dataset import loadJava, loadOJ, javaParser
# from model import model,losses
from gensim.models.word2vec import Word2Vec, LineSentence
from tqdm import tqdm
import dataset
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from dataset.javaParser import pipline,gcjPipline
from dataset.cParser import cloPipline,claPipline
from model.model import CLO,CLA,JKNet
from model.losses import supcon
from  util import split_dataset
import torch as th
import torch.nn as nn
L2_PENALTY = 0.0005

def node2emb(nodeInfo, keyword='gcj'):
    nodeemb = []
    embSavePath = 'data/emb/' + keyword + '_w2v_128'
    word2vec = Word2Vec.load(embSavePath).wv
    vocab = word2vec.vocab
    if type(nodeInfo).__name__ == 'dict':
        for k, v in nodeInfo.items():
            str = v.replace("\n", "")
            list = str.split()
            # print(word2vec[list[0]])
            if list[0] not in vocab:
                print(nodeInfo,"======",list[0])
            nodeemb.append(word2vec[list[0]])
    else:
        if type(nodeInfo).__name__ == 'list':
            for v in nodeInfo:
                str = v.replace("\n", "")
                list = str.split()
                if list[0] not in vocab:
                    print(nodeInfo, "======", list[0])
                    print('v',v)
                    print('str',str)
                nodeemb.append(word2vec[list[0]])
    return nodeemb



def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e',
                        help='number of epochs to train',
                        type=int, default=100)
    parser.add_argument('--batch_size', '-b',
                        help='number of batch_size to train',
                        type=int, default=16)
    parser.add_argument('--learning_rate', '-lr',
                        help='Learning rate',
                        type=float, default=0.0001)
    parser.add_argument('--n_layers', '-nl',
                        help='Number of convolution layers',
                        type=int, default=2)
    parser.add_argument('--n-units', '-nu',
                        help='Size of middle layers.',
                        type=int, default=128)
    parser.add_argument('--train_ratio', '-tr',
                        help='Ratio of train sets.',
                        type=int, default=8)
    parser.add_argument('--val_ratio', '-vr',
                        help='Ratio of val sets.',
                        type=int, default=1)
    parser.add_argument('--func_path', '-fp',
                        help='The path to source code dataset',
                        type=str,
                        default='data/oj_funcs.pkl')
    parser.add_argument('--clone_pair_path', '-cpp',
                        help='The path to source code dataset',
                        type=str,
                        default='data/oj_pair_ids.pkl')
    parser.add_argument('--datasetName', '-dn',
                        help='The name to the datasetName',
                        type=str,choices=('oj', 'gcj'),
                        default='oj')
    parser.add_argument('--model_path', '-mp',
                        help='The path to save model',
                        type=str,
                        default='model/model.pth')


    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    clone = False

    if args.datasetName == 'gcj':
        parserCode = gcjPipline
        Dataset = loadJava.java_DATASET( file_path=args.func_path, clone_path=args.clone_pair_path,clone = clone)
    else:
        if args.datasetName == 'oj':
            Dataset = loadOJ.OJ_DATASET(file_path=args.func_path, clone_path=args.clone_pair_path,clone=clone)
            parserCode = claPipline
        else:
            raise Exception("wrong dataName!")
    model = JKNet(args.n_units,n_layers=args.n_layers, n_units=args.n_units).to(device)
    shuffle = True
    train_loader, validate_loader, test_loader = split_dataset(Dataset,shuffle,args.batch_size,args.train_ratio,args.val_ratio)


    optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate)


    for epoch in range(1, args.epochs + 1):
        model.train()
        for i, item in enumerate(tqdm(train_loader)):
            index, code, labels = item
            print('index',index)
            torch.save(index, 'index.pt')
            torch.save(code,'code.pt')
            nodeInfo, srcSg, srcEdfg, dstSg, dstEdfg, nodeNum, sgInfo, edfgInfo = parserCode(code,dataSetName=args.datasetName)
            Feat = th.FloatTensor(node2emb(nodeInfo,args.datasetName)).to(device)
            sg = dgl.graph((srcSg, dstSg)).to(device)
            edfg = dgl.graph((srcEdfg, dstEdfg)).to(device)
            sgFeat = th.FloatTensor(node2emb(sgInfo,args.datasetName)).to(device)
            edfgFeat = th.FloatTensor(node2emb(edfgInfo,args.datasetName)).to(device)
            # res =  model(nodeNum, sg, edfg, Feat, sgFeat, edfgFeat)
            # loss = supcon(res,labels)
            # th.save(model.state_dict(), args.model_path)
            # loss.requires_grad_(True)
            # print('loss',loss)
            # loss.backward()
            # optimizer.step()
            # break

        model.eval()
        with torch.no_grad():
            for i, item in enumerate(tqdm(test_loader)):
                index, code, labels = item
                nodeInfo, srcSg, srcEdfg, dstSg, dstEdfg, nodeNum, sgInfo, edfgInfo = parserCode(code,
                                                                                                 dataSetName=args.datasetName)
                Feat = th.FloatTensor(node2emb(nodeInfo, args.datasetName)).to(device)
                sg = dgl.graph((srcSg, dstSg)).to(device)
                edfg = dgl.graph((srcEdfg, dstEdfg)).to(device)
                sgFeat = th.FloatTensor(node2emb(sgInfo, args.datasetName)).to(device)
                edfgFeat = th.FloatTensor(node2emb(edfgInfo, args.datasetName)).to(device)
                res = model(nodeNum, sg, edfg, Feat, sgFeat, edfgFeat)
                loss = supcon(res, labels)
                print('test loss: ', loss)
                # break

    th.save(model.state_dict(), args.model_path)



if __name__ =='__main__':
    main()
