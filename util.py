import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
def split_dataset(dataset,shuffle,batch_size,train_ratio,val_ratio):

    train_size = int(len(dataset) *(train_ratio/10.0))
    validate_size = int(len(dataset) * (val_ratio/10.0))
    test_size = len(dataset) - validate_size - train_size
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                  [train_size, validate_size,
                                                                                   test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return train_loader, validate_loader, test_loader

def load_model(params_small,model_large):
    params_large = model_large.state_dict()
    for k,v in params_small.items():
        layer_name = 'jknet.'+k
        if layer_name in params_large.keys():
            params_large[layer_name] = params_small[k]
        model_large.load_state_dict(params_large)

def nor(data):
    data_max, _ = torch.max(data, dim=1, keepdim=True)
    data_min, _ = torch.min(data, dim=1, keepdim=True)
    data = (data - data_min.detach()) / (data_max - data_min)
    return  data

# def node2emb(nodeInfo, keyword='gcj'):
#     nodeemb = []
#     embSavePath = 'data/emb/' + keyword + '_w2v_128'
#     word2vec = Word2Vec.load(embSavePath).wv
#     if type(nodeInfo).__name__ == 'dict':
#         print(nodeInfo.values())
#         for k, v in nodeInfo.items():
#             str = v.replace("\n", "")
#             list = str.split()
#             nodeemb.append(word2vec[list[0]])
#     else:
#         if type(nodeInfo).__name__ == 'list':
#             for v in nodeInfo:
#                 str = v.replace("\n", "")
#                 list = str.split()
#                 nodeemb.append(word2vec[list[0]])
#     return nodeemb
from flair.embeddings import WordEmbeddings
from flair.data import Sentence
glove_embedding = WordEmbeddings('glove')
def node2emb(nodeInfo, keyword='gcj'):
    nodeemb,nodeInfotemp=[],[]
    if type(nodeInfo).__name__ == 'dict':
        for i in nodeInfo.keys():
            nodeInfotemp.append(nodeInfo[i])
        c = nodeInfotemp
        sentence = Sentence(c)
        with torch.no_grad():
            torch.cuda.empty_cache()
            glove_embedding.embed(sentence)
        for token in sentence:
            nodeemb.append(token.embedding.cpu().numpy())
    else:
        if type(nodeInfo).__name__ == 'list':
            c = nodeInfo
            sentence = Sentence(c)
            with torch.no_grad():
                torch.cuda.empty_cache()
                glove_embedding.embed(sentence)
            for token in sentence:
                nodeemb.append(token.embedding.cpu().numpy())
    return nodeemb

if __name__ == '__main__':
    split_dataset()