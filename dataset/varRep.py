import javalang
from javalang.ast import Node
import pandas as pd
from pycparser import c_parser
import copy
from dataset.Node import ASTNode,SingleNode
import numpy as np

'''
使用的变量的解释
ID：每个节点的ID，从0开始
depth: 每个节点所在的层次，从0开始
nodeInfo:  存储所有node信息，key:nodetype, value:ID
varName2info：  变量定义的时候变量名：id:【变量名，变量类型，层】，id是唯一的因此作为key
varDec = False ：当前节点是不是一个变量声明
memRef = False ：当前节点是不是一个变量的引用

'''
ID=0
depth = 0
varDec = False
memRef = False
forPar = False
varName2info = {}
nodeInfo = {}
nodeInfoRep ={}
edfgEdgSrc = []
edfgEdgDst = []
edfgEdgInfo = []

sgEdgSrc = []
sgEdgDst = []
sgEdgInfo=[]

def nodeName2Type():
    #对之前已经存储过的错误类型进行更正
    for k,v in nodeInfoRep.items():
        nodeInfo[k] = v

def searchDec(memRef,id,depth):
    '''
    查找的条件：层次信息和ID最近的
    '''
    distance=9999999999
    tempId = -1
    for k,v in varName2info.items():
        if v[0] == memRef:
            if v[2]< depth:
                tempId = k if id - k < distance else tempId
    print(tempId,memRef)
    if tempId > 0:
        '''
        如果同时定义两个变量，则就是需要遍历已经存的nodeInfo
        547 ['b2', 'int', 6]
        549 ['b3', 'b2', 6]
        '''
        need=True
        tempType = varName2info[tempId][1]
        while(need):
            replace = False
            for k,v in varName2info.items():
                if v[1] != 'ReferenceType'and v[0] == tempType  :
                    replace = True
                    tempId = k
                    tempType = v[1]
            if replace == False:
                need = False

        print(memRef + str(id) + " type is " + tempType + " according to " + str(tempId))
        nodeInfoRep[id] = tempType
        edfgEdgSrc.append(tempId)
        edfgEdgDst.append(id)


def checkDec(token,subToken):
    '''
    :param token: 节点的type methodeDeclaration
    :param subToken: 之后判断他是一个Declarator 还是memberReference
    :return:
    '''
    if token == subToken:
        return True
    else:
        return False



def java_get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token

def java_get_children(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item
    return list(expand(children))


def java_trans_path(node,ast_path_single,ast_path,depth=0,parentType=''):
    global ID,varDec, memRef,nodeInfo,varName2info,forPar
    parentID=ID
    token, children = java_get_token(node), java_get_children(node)
    nodeInfo[ID] = token


    if(checkDec(token,'VariableDeclarator')):
        parentType = 'VariableDeclarator'
    else:
        if(checkDec(token,"MemberReference")):
            parentType = "MemberReference"
        else:
            if (checkDec(token,'FormalParameter')):
                parentType = 'FormalParameter'

    varDec = False
    memRef = False
    forPar = False
    if(parentType == 'VariableDeclarator'):
        varDec = True
    else:
        if(parentType == "MemberReference"):
            memRef = True
        else:
            if(parentType == 'FormalParameter'):
                forPar = True

    ast_path_single.append(str(ID)+token+"  "+str(depth))
    ID+=1
    if len(children)==0:
        ast_path.append(copy.deepcopy(ast_path_single))



    for child in children:
        #建立边关系
        sgEdgSrc.append(parentID)
        sgEdgDst.append(ID)
        sgEdgInfo.append('parent')

        sgEdgSrc.append(ID)
        sgEdgDst.append(parentID)
        sgEdgInfo.append('child')

        if forPar:
            print(ID,child,type(child))
        if isinstance(child,str):
            if memRef:
                searchDec(child,ID,depth)
            if varDec:
                varName2info[ID] = [child,nodeInfo[ID-2],depth]
            if forPar:
                varName2info[ID] = [child, nodeInfo[ID - 1], depth]
        java_trans_path(child, ast_path_single,ast_path,depth+1,parentType)
        ast_path_single.pop()

def get_java_ast(code,dataSetName = 'gcj'):
    if dataSetName == 'bcb':
        tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
    if dataSetName == 'gcj':
        tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse()
    return tree


def java_trans_path_list(code,dataSetName='gcj'):
    ast=get_java_ast(code,dataSetName)
    ast_path_single=[]
    ast_path=[]
    java_trans_path(ast, ast_path_single,ast_path)
    for i in ast_path:
        print(i)
    return ast_path

def word2vecTest():
    import gensim
    corpus = list(nodeInfo.values())

    corpus_str = ' '.join(c for c in corpus)
    from gensim.models.word2vec import Word2Vec
    w2v = Word2Vec(corpus, size=128, workers=16, sg=1, min_count=3)
    w2v.save('node_w2v_128')


    word2vec = Word2Vec.load('node_w2v_128').wv
    vocab = word2vec.vocab
    max_token = word2vec.vectors.shape[0]
    print(word2vec)
    print("-------------1111111----------------")
    print(vocab)
    print("---------------22222--------------")
    print(max_token)

    embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors
    print(embeddings.shape)

def getGraph(dataSetName,dataSetPath):
    data = pd.read_pickle(dataSetPath)
    if dataSetName == 'bcb':
        for i ,j in data.iterrows():
            parserBcb()
    if dataSetName == 'gcj':
        for i ,j in data.iterrows():
            parserGcj()
    if dataSetName == 'oj':
        for i ,j in data.iterrows():
            parserOj()

import pandas as pd
if  __name__ == '__main__':
    bcb_path = '../data/gcj_funcs.pkl'
    data = pd.read_pickle(bcb_path)
    print(data)
    temp=0
    for i,j in data.iterrows():
        print(j['code'])
        code = j["code"]
        temp+=1
        if temp==1:
            break
    # code = 'public class test{int a=4;string s = "a"; public void f1(){int b=0;b=9;s="op"; if(true){a=-1;}else{a=1;}} }'
    java_trans_path_list(code)

    nodeName2Type()
    print("================")
    word2vecTest()
    # for k, v in nodeInfo.items():
    #     print(k, v)
    # print(edfgEdg)
    print(len(sgEdgSrc),len(sgEdgDst))



