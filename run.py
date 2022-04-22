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
from dataset.javaParser import pipline
from dataset.cParser import cloPipline,claPipline
from model.model import CLO,CLA
import torch as th

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
            nodeemb.append(word2vec[list[0]])
    else:
        if type(nodeInfo).__name__ == 'list':
            for v in nodeInfo:
                str = v.replace("\n", "")
                list = str.split()
                nodeemb.append(word2vec[list[0]])
    return nodeemb

def CrossEntropyLoss_label_smooth(outputs, targets,device, num_classes=104, epsilon=0.1):
    N = targets.size(0)
    if device=='cuda':
        smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1)).cuda()
    else:
        smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1))
    targets = targets.data
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1 - epsilon)
    log_prob = F.log_softmax(outputs, dim=1)
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss

def train(graph, model, xs, idx_train, ts_train, optimizer):
    model.train()
    optimizer.zero_grad()
    ys = F.log_softmax(model(graph, xs), dim=1)
    loss = F.nll_loss(ys[idx_train], ts_train)
    loss.backward()
    optimizer.step()

def evaluate(graph, model, xs, idx_test, ts_test):
    model.eval()
    with torch.no_grad():
        ys = model(graph, xs)[idx_test]
        predict = ys.max(1, keepdim=True)[1]
        n_correct = predict.eq(ts_test.view_as(predict)).sum().item()
        accuracy = n_correct / ts_test.shape[0]
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e',
                        help='number of epochs to train',
                        type=int, default=100)
    parser.add_argument('--batch_size', '-b',
                        help='number of batch_size to train',
                        type=int, default=8)
    parser.add_argument('--learning_rate', '-lr',
                        help='Learning rate',
                        type=float, default=0.0005)
    parser.add_argument('--n-layers',
                        help='Number of convolution layers',
                        type=int, default=4)
    parser.add_argument('--n-units',
                        help='Size of middle layers.',
                        type=int, default=128)
    parser.add_argument('--n-classes',
                        help='the number of classes.',
                        type=int, default=1)
    parser.add_argument('--cloneChoices',
                        help='clone detection:0, classification:1',
                        type=int, default=0)
    parser.add_argument('--func_path',
                        help='The path to source code dataset',
                        type=str,
                        default='data/bcb_funcs.pkl')
    parser.add_argument('--clone_pair_path',
                        help='The path to the clone_pair file',
                        type=str,
                        default='data/bcb_pair_ids.pkl')
    parser.add_argument('--datasetName',
                        help='The name to the datasetName',
                        type=str,choices=('oj', 'gcj', 'bcb'),
                        default='bcb')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.cloneChoices == 0:
        clone = True
        model = CLO(args.n_units, args.n_classes).to(device)
    else:
        clone = False
        model = CLA(args.n_units,args.n_classes,n_units=args.n_units).to(device)

    if args.datasetName == 'gcj' or args.datasetName == 'bcb':
        parserCode = pipline
        Dataset = loadJava.java_DATASET( file_path=args.func_path, clone_path=args.clone_pair_path)
    else:
        if args.datasetName == 'oj':
            Dataset = loadOJ.OJ_DATASET(file_path=args.func_path, clone_path=args.clone_pair_path,clone=clone)
            if clone:
                parserCode = cloPipline
            else:
                parserCode = claPipline
    dataloader = DataLoader(Dataset, batch_size=args.batch_size, shuffle=True)


    optimizer = th.optim.Adam(model.parameters(), lr= args.learning_rate)
    precision, recall, f1 = 0, 0, 0
    predicts = []
    trues = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        predicts1 = []
        trues1 = []
        predicts2 = []
        trues2 = []
        predicts3 = []
        trues3 = []
        predicts4 = []
        trues4 = []
        predicts5 = []
        trues5 = []
        for i, item in enumerate(tqdm(dataloader)):
            if clone:
                loss_function = th.nn.BCELoss()
                index, code1, code2, labels = item
                nodeInfo1, nodeInfo2, srcSg1, srcEdfg1, dstSg1, dstEdfg1, nodeNum1, \
                srcSg2, srcEdfg2, dstSg2, dstEdfg2, \
                nodeNum2, sgInfo1, edfgInfo1, sgInfo2, edfgInfo2 = parserCode(code1, code2, dataSetName=args.datasetName)
                Feat1 = th.FloatTensor(node2emb(nodeInfo1,args.datasetName)).to(device)
                Feat2 = th.FloatTensor(node2emb(nodeInfo2,args.datasetName)).to(device)
                sg1 = dgl.graph((srcSg1, dstSg1)).to(device)
                edfg1 = dgl.graph((srcEdfg1, dstEdfg1)).to(device)
                sg2 = dgl.graph((srcSg2, dstSg2)).to(device)
                edfg2 = dgl.graph((srcEdfg2, dstEdfg2)).to(device)
                sgFeat1 = th.FloatTensor(node2emb(sgInfo1,args.datasetName)).to(device)
                sgFeat2 = th.FloatTensor(node2emb(sgInfo2,args.datasetName)).to(device)
                edfgFeat1 = th.FloatTensor(node2emb(edfgInfo1,args.datasetName)).to(device)
                edfgFeat2 = th.FloatTensor(node2emb(edfgInfo2,args.datasetName)).to(device)
                res = model(nodeNum1, sg1, edfg1, Feat1, sgFeat1, edfgFeat1,
                      nodeNum2, sg2, edfg2, Feat2, sgFeat2, edfgFeat2)
                # labels = torch.tensor(labels)
                labels_list = []
                for item in labels:
                    if item:
                        labels_list.append(1)
                    else:
                        labels_list.append(0)
                # print('=' * 80)

                labels_store = labels
                labels= th.FloatTensor(labels_list)
                labels = labels.view(-1, 1)
                loss = loss_function(res, labels)
                # print('labels', labels_store)
                # predicted = (res.data > 0.5).cpu().numpy()
                # predicts.extend(predicted)
                # trues.extend(labels.cpu().numpy())
                # num = 0
                # for i in labels_store:
                #     if i == 1:
                #         trues1.extend(trues[num])
                #         predicts1.extend(predicts[num])
                #     elif i==2:
                #         trues2.extend(trues[num])
                #         predicts2.extend(predicts[num])
                #     elif i==3:
                #         trues3.extend(trues[num])
                #         predicts3.extend(predicts[num])
                #     elif i==4:
                #         trues4.extend(trues[num])
                #         predicts4.extend(predicts[num])
                #     elif i==5:
                #         trues5.extend(trues[num])
                #         predicts5.extend(predicts[num])
                #     num += 1
                # 
                # print('predictes',predicts,'\n',predicts1,predicts2,predicts3,predicts4,predicts5)
                # print('true',trues,'\n',trues1,trues2,trues3,trues4,trues5)
                # p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
                # print(p,r,f)

            else:
                loss_function = CrossEntropyLoss_label_smooth
                index, code, labels = item
                nodeInfo, srcSg, srcEdfg, dstSg, dstEdfg, nodeNum, sgInfo, edfgInfo = claPipline(code)
                Feat = th.FloatTensor(node2emb(nodeInfo,args.datasetName)).to(device)
                sg = dgl.graph((srcSg, dstSg)).to(device)
                edfg = dgl.graph((srcEdfg, dstEdfg)).to(device)
                sgFeat = th.FloatTensor(node2emb(sgInfo,args.datasetName)).to(device)
                edfgFeat = th.FloatTensor(node2emb(edfgInfo,args.datasetName)).to(device)
                # print('sg',len(srcSg),len(dstSg),len(srcEdfg),len(dstEdfg),len(sgInfo),len(edfgInfo))
                res = model(nodeNum, sg, edfg, Feat, sgFeat, edfgFeat)
                _, predicted = torch.max(res.data, 1)
                # print('res',predicted)
                # print('label',labels)
                loss = loss_function(res, labels,device)
            print(loss)
            loss.backward()
            optimizer.step()

        #evaluate
        #clone
        predicts1 = []
        trues1 = []
        predicts2 = []
        trues2 = []
        predicts3= []
        trues3 = []
        predicts4 = []
        trues4 = []
        predicts5 = []
        trues5 = []

        #cla
        total_acc = 0.0
        total = 0.0
        
        model.eval()
        for i, item in enumerate(tqdm(dataloader_eval)):
            if clone:#克隆的evalute
                with torch.no_grad():
                    index, code1, code2, labels = item
                    nodeInfo1, nodeInfo2, srcSg1, srcEdfg1, dstSg1, dstEdfg1, nodeNum1, \
                    srcSg2, srcEdfg2, dstSg2, dstEdfg2, \
                    nodeNum2, sgInfo1, edfgInfo1, sgInfo2, edfgInfo2 = parserCode(code1, code2,
                                                                                  dataSetName=args.datasetName)
                    Feat1 = th.FloatTensor(node2emb(nodeInfo1, args.datasetName)).to(device)
                    Feat2 = th.FloatTensor(node2emb(nodeInfo2, args.datasetName)).to(device)
                    sg1 = dgl.graph((srcSg1, dstSg1)).to(device)
                    edfg1 = dgl.graph((srcEdfg1, dstEdfg1)).to(device)
                    sg2 = dgl.graph((srcSg2, dstSg2)).to(device)
                    edfg2 = dgl.graph((srcEdfg2, dstEdfg2)).to(device)
                    sgFeat1 = th.FloatTensor(node2emb(sgInfo1, args.datasetName)).to(device)
                    sgFeat2 = th.FloatTensor(node2emb(sgInfo2, args.datasetName)).to(device)
                    edfgFeat1 = th.FloatTensor(node2emb(edfgInfo1, args.datasetName)).to(device)
                    edfgFeat2 = th.FloatTensor(node2emb(edfgInfo2, args.datasetName)).to(device)
                    res = model(nodeNum1, sg1, edfg1, Feat1, sgFeat1, edfgFeat1,
                                nodeNum2, sg2, edfg2, Feat2, sgFeat2, edfgFeat2)
                    

                    #评价指标
                    labels_store = labels
                    labels_list = []
                    for item in labels:
                        if item:
                            labels_list.append(1)
                        else:
                            labels_list.append(0)
                    # print('=' * 80)

                    labels_store = labels
                    labels = th.FloatTensor(labels_list)
                    predicted = (res.data > 0.5).cpu().numpy()
                    predicts.extend(predicted)
                    trues.extend(labels.cpu().numpy())
                    num = 0
                    for i in labels_store:
                        if i == 1:
                            trues1.extend(trues[num])
                            predicts1.extend(predicts[num])
                        elif i == 2:
                            trues2.extend(trues[num])
                            predicts2.extend(predicts[num])
                        elif i == 3:
                            trues3.extend(trues[num])
                            predicts3.extend(predicts[num])
                        elif i == 4:
                            trues4.extend(trues[num])
                            predicts4.extend(predicts[num])
                        elif i == 5:
                            trues5.extend(trues[num])
                            predicts5.extend(predicts[num])
                        num += 1
                    # print('predictes', predicts, '\n', predicts1, predicts2, predicts3, predicts4, predicts5)
                    # print('true', trues, '\n', trues1, trues2, trues3, trues4, trues5)
            else:
                with torch.no_grad():
                    #model
                    index, code, labels = item
                    nodeInfo, srcSg, srcEdfg, dstSg, dstEdfg, nodeNum, sgInfo, edfgInfo = claPipline(code)
                    Feat = th.FloatTensor(node2emb(nodeInfo, args.datasetName)).to(device)
                    sg = dgl.graph((srcSg, dstSg)).to(device)
                    edfg = dgl.graph((srcEdfg, dstEdfg)).to(device)
                    sgFeat = th.FloatTensor(node2emb(sgInfo, args.datasetName)).to(device)
                    edfgFeat = th.FloatTensor(node2emb(edfgInfo, args.datasetName)).to(device)
                    res = model(nodeNum, sg, edfg, Feat, sgFeat, edfgFeat)
                    # calc training acc
                    _, predicted = torch.max(logits.data, 1)
                    total_acc += (predicted == labels).sum()
                    total += len(labels)
        if clone:
            if args.datasetName == 'bcb':
                if dataset == 'bcb':
                    weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
                    p1, r1, f1, _ = precision_recall_fscore_support(trues1, predicts1, average='binary')
                    print("testing results(P,R,F1):Type-" + str(1) + ": " + str(p1) + " " + str(r1) + " " + str(f1))
                    p2, r2, f2, _ = precision_recall_fscore_support(trues2, predicts2, average='binary')
                    print("testing results(P,R,F1):Type-" + str(2) + ": " + str(p2) + " " + str(r2) + " " + str(f2))
                    p3, r3, f3, _ = precision_recall_fscore_support(trues3, predicts3, average='binary')
                    print("testing results(P,R,F1):Type-" + str(3) + ": " + str(p3) + " " + str(r3) + " " + str(f3))
                    p4, r4, f4, _ = precision_recall_fscore_support(trues4, predicts4, average='binary')
                    print("testing results(P,R,F1):Type-" + str(4) + ": " + str(p4) + " " + str(r4) + " " + str(f4))
                    p5, r5, f5, _ = precision_recall_fscore_support(trues5, predicts5, average='binary')
                    print("testing results(P,R,F1):Type-" + str(5) + ": " + str(p5) + " " + str(r5) + " " + str(f5))
                    p = weights[1] * p1+weights[2] * p2+weights[3] * p3+weights[4] * p4+weights[5] * p5
                    r = weights[1] * r1+weights[2] * r2+weights[3] * r3+weights[4] * r4+weights[5] * r5
                    f1 = weights[1] * f1+weights[2] * f2+weights[3] * f3+weights[4] * f4+weights[5] * f5
                    print("Total testing results(P,R,F1):"  + str(p) + " " + str(r) + " " + str(f1))
                else:
                    precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
                    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))
        else:
            print("epoch: {}, Testing results(Acc):{}".format(epoch+1,total_acc / total))


if __name__ == '__main__':
    main()
