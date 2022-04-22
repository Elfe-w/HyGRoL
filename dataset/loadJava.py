from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
from javaParser import pipline,gcjPipline
'''
每次可以返回一批次的的解析代码
'''

class java_DATASET(Dataset):
    def __init__(self,wrongData=[],dataSetName = 'gcj',file_path='../data/gcj_funcs.pkl',clone_path = '../data/gcj_pair_ids.pkl',clone=True):
        self.clone = clone
        self.file_path = file_path
        self.clone_path = clone_path
        self.data = pd.read_pickle(self.file_path)
        self.clonePair = pd.read_pickle(self.clone_path)
        if self.clone:
            self.len = len(self.clonePair)
        else:
            self.len = len(self.data)
        if dataSetName == 'bcb':
            self.wrongData = [6430, 6425, 6397, 6400, 6400, 6402, 6399, 6399, 6397, 6423, 6412, 6436, 6414, 6426, 6432, 6426, 6426, 6429, 6402, 6430]
        else:
            self.wrongData = []
        self.code = self.data.loc[:, 'code'].tolist()
        self.label = self.data.loc[:, 'file_label'].tolist()
        # self.code = self.data.loc[:,'code'].tolist()
        # self.label =self.data.loc[:,'file_label'].tolist()
        self.cloneLabel = self.clonePair.loc[:,'label'].tolist()
    def print_test(self,keyword):
        print('print',keyword)
    def __getitem__(self,index):
        if self.clone == False:
            code = self.code[index]
            label = self.label[index]
            return index, code,label-1
        else:
            if index in self.wrongData:
                return -1,'nan','nan',[-1]
            c1Index = self.clonePair.loc[index,'id1']
            c2Index = self.clonePair.loc[index,'id2']
            label = self.clonePair.loc[index,'label']
            code1 = self.data.loc[self.data['id']==c1Index,'code']
            code2 = self.data.loc[self.data['id']==c2Index,'code']
            for index, value in code1.items():
                code1 = value
                break
            for index, value in code2.items():
                code2 = value
                break
            # print('code1',code1)
            # print('code2',code2)
            #print('label',type(label),label,'code1type',type(code1),'code2type',type(code2),'pairID',index,'c1index',c1Index,'c2index',c2Index)
            # if type(code1).__name__ == 'Series' or type(code2).__name__ == 'Series' :
            #     self.wrongData.append(index)
            #     print(self.wrongData)
            #     #获取到dataset之后判断如果全是-1就删除掉
            #     return -1,-1,-1,[-1]
            return index,code1,code2,label

    def __len__(self):
        return self.len

if __name__ == '__main__':
    from tqdm import tqdm

    data = pd.read_pickle('../data/gcj_funcs.pkl')
    print(data)
    javaDataset = java_DATASET(file_path='../data/gcj_funcs.pkl',clone_path='../data/gcj_pair_ids.pkl',clone=False)
    dataloader = DataLoader(javaDataset, batch_size=128, shuffle=True)
    # file_path = '../data/oj_funcs.pkl'
    # alldata = pd.read_pickle(file_path)
    for i, item in enumerate(tqdm(dataloader)):
        index, code, label = item
        print('step is {}, index is {}'.format(i, index))
        # print(type(code1),len(code1))
        # print(type(code2),len(code2))
        gcjPipline(code,dataSetName = 'gcj')
        break
    # print(wrongData)


