from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
from cParser import claPipline,cloPipline


class OJ_DATASET(Dataset):
    def __init__(self,file_path='../data/oj_funcs.pkl',clone_path = '../data/oj_pair_ids.pkl',clone=False):
        self.clone = clone
        self.file_path = file_path
        self.clone_path = clone_path
        self.data = pd.read_pickle(self.file_path)
        self.clonePair = pd.read_pickle(self.clone_path)
        if self.clone:
            self.len = len(self.clonePair)
        else:
            self.len = len(self.data)
        self.code = self.data.loc[:,'code'].tolist()
        self.label =self.data.loc[:,'file_label'].tolist()

    def print_test(self,keyword):
        print('print',keyword)


    def __getitem__(self,index):
        if self.clone == False:
            code = self.code[index-1]
            label = self.label[index-1]
            return index, code,label-1
        else:
            c1Index = self.clonePair.loc[index, 'id1']
            c2Index = self.clonePair.loc[index, 'id2']
            label = self.clonePair.loc[index, 'label']
            code1 = self.data.loc[self.data['id'] == c1Index, 'code']
            code2 = self.data.loc[self.data['id'] == c2Index, 'code']
            for index, value in code1.items():
                code1 = value
                break
            for index, value in code2.items():
                code2 = value
                break
            # print('code1',code1)
            # print('code2',code2)
            #print('label',type(label),label,'code1type',type(code1),'code2type',type(code2),'pairID',index,'c1index',c1Index,'c2index',c2Index)
            return index, code1, code2, label

    def __len__(self):
        return self.len

if __name__ == '__main__':
    from tqdm import tqdm
    clone = False

    ojDataset = OJ_DATASET(clone=clone)
    dataloader = DataLoader(ojDataset, batch_size=2, shuffle=True)
    # print(len(dataloader))
    if clone:
        for i, item in enumerate(tqdm(dataloader)):
            index, code1,code2, label = item
            print('step is {}, index is {}'.format(i, index))
            # print('index is {},data is {},\nlabel is {}'.format(index,data, label))
            # print(type(data))
            cloPipline(code1,code2)
    else:
        for i, item in enumerate(tqdm(dataloader)):
            index, data, label = item
            print('step is {}, index is {}'.format(i, index))
            print('index is {},data is {},\nlabel is {}'.format(index,data, label))
            print(type(data))
            allNodeInfo,allSgSrc,allSgDst,allEdfgSrc,allEdfgDst = claPipline(data)
            print('allNodeInfo: ',len(allNodeInfo),allNodeInfo)
            print('allSgSrc',len(allSgSrc),len(allSgDst))
            print(allSgSrc)
            print(allSgDst)
            print('allEdfgSrc', len(allEdfgSrc),len(allEdfgDst))
            print(allEdfgSrc)
            print(allEdfgDst)
            break

