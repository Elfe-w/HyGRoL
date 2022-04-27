import gensim
import pandas as pd
import os
from cParser import parserCForCorpus
from javaParser import parserJavaForCorpus
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec, LineSentence
import gensim
import numpy as np
def save_corpus(dataPath,saveRootPath,keyword):
    data = pd.read_pickle(dataPath)
    totalNum = len(data)
    with tqdm(data.iterrows(), desc='generate '+keyword+' corpus...') as tbar:
        for i, j in tbar:
            tbar.set_postfix(ratio =i / totalNum, id=i)
            tbar.update()  # 默认参数n=1，每update一次，进度+n
            code = j['code']
            if keyword == 'oj':
                 nodeInfo = parserCForCorpus(code)
            else:
                nodeInfo = parserJavaForCorpus(code,keyword)
            corpus = list(nodeInfo.values())
            corpus_str = ' '.join(c.replace("\n","") for c in corpus)
            if not os.path.exists(saveRootPath):
                os.mkdir(saveRootPath)
            with open(saveRootPath+keyword+'_corpus.txt','a',encoding='utf-8') as f:
                f.write(corpus_str)
                f.write('\n')
            #最后还要在语料库里面添加边上的标签数据
            corpus = ['parent', 'child', 'route', 'value']
            corpus_str = ' '.join(c.replace("\n", "") for c in corpus)
            with open(saveRootPath+keyword+'_corpus.txt','a',encoding='utf-8') as f:
                f.write(corpus_str)
                f.write('\n')
def word2emb(corpusPath,embSavePath,keyword):
    corpus =LineSentence(corpusPath)
    w2v = Word2Vec(corpus, size=128, workers=16, sg=1, min_count=1)
    path = embSavePath+keyword+'_w2v_128'
    w2v.save(path)
def pipline(keyword):
    '''
    先获得整个待解析代码
    然后进行代码解析
    解析后的内容写到TXT文件中
    获取完语料之后进行编码
    :return:
    '''
    '''
    1.获得语料库
    '''
    dataPath = '../data/'+keyword+'_funcs.pkl'
    print(dataPath)
    saveRootPath = '../data/emb/'
    save_corpus(dataPath, saveRootPath, keyword)
    '''
    2.保存编码
    '''
    corpusPath='../data/emb/'+keyword+'_corpus.txt'
    embSavePath = '../data/emb/'
    with open(corpusPath, 'r', encoding='utf-8') as f:
        print(keyword)
    word2emb(corpusPath,embSavePath,keyword)
    '''
    *3.读取编码测试
    '''
    word2vec = Word2Vec.load(embSavePath+keyword+'_w2v_128').wv
    vocab = word2vec.vocab
    print('-'*20,'编码输出测试','-'*20)
    print(vocab['value'])
    print(vocab['route'])
    print(vocab['parent'])
    print(vocab['child'])


if __name__ == '__main__':
    pipline('oj')
    pipline('gcj')
    pipline('bcb')
    # #Enumerator
    # embSavePath = '../data/emb/oj_w2v_128'
    # word2vec = Word2Vec.load(embSavePath).wv
    # vocab = word2vec.vocab
    # if 'value' in vocab:
    #     print('000')
    # else:
    #     print('777')
    # print('-' * 20, '编码输出测试', '-' * 20)
    # print(word2vec['value'])
    # print(vocab['route'])
    # print(vocab['parent'])
    # print(vocab['child'])
    # print(vocab['Enumerator'])
    # path = '../data/emb/gcj_w2v_128'
    # word2vec = Word2Vec.load(path).wv
    # vocab = word2vec.vocab
    # max_token = word2vec.vectors.shape[0]
    #
    # print(word2vec)
    # print("-------------1111111----------------")
    # print('vocab',vocab)
    # print("---------------22222--------------")
    # print('max_token',max_token)
    #
    # embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
    # embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors
    # print('embedding shape: ',embeddings.shape)
    # result = [vocab['CompilationUnit'].index if 'CompilationUnit' in vocab else max_token]
    # print(result)
    # print(vocab['CompilationUnit'])
    # print(vocab['PackageDeclaration'])
    # print('getVector: ', word2vec.get_vector('CompilationUnit'))


