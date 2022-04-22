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
lastId = -1

depth = 0
edfgEdgSrc = []
edfgEdgDst = []
edfgEdgInfo = []

#处理分支结构,可能出现嵌套的情况，因此类似括号匹配，用栈去做
ifFlag = []
ifCon = []
ifConId = [-1]
ifDepth = [-999]

ifSrc = []
ifDst = []
dataSrc = []
dataDst = []



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


def dataEdge(node):
    '''
    :param node: 传入一个节点
    :return: 返回从该节点出发第一个孩子下层次有多深
    '''
    children = java_get_children(node)
    if len(children) == 0:
        return 1
    else:
        for child in children:
           return  dataEdge(child)+1



def java_trans_path(node,ast_path_single,ast_path,depth=0,parentType=''):
    global ID,lastId,ifFlag,ifDepth,ifCon,ifConId,dataDst,dataSrc


    token, children = java_get_token(node), java_get_children(node)
    ast_path_single.append(str(ID)+token+"  "+str(depth))


    if depth <= ifDepth[-1]:
        ifFlag.pop(-1)
        ifCon.pop(-1)
        ifDepth.pop(-1)
        ifConId.pop(-1)
    if token == 'IfStatement':
        ifFlag.append(True)
        ifDepth.append(depth)
        ifCon.append(False)
    if token == 'Assignment':
        #直接通过递归遍历创建边
        count = 0
        for child in children:
            if count == 0:
                dataDst.append(ID + dataEdge(child))
            else:
                if count == 1:
                    dataSrc.append(dataDst[-1] + dataEdge(child)-1)
                else:
                    break
            count+=1

    parentType = token
    if depth == ifDepth[-1]+1 and ifFlag[-1] :#确定是ifstatement下 的一个节点
        if token == 'BlockStatement' or token == 'StatementExpression':#如果是个分支节点,如果带大括号他就是个BlockStatement如果没有大括号而是单个语句，他就是个StatementExpression
            if not ifCon[-1]:
                ifCon[-1] = True
                ifConId.append(lastId)
            ifSrc.append(ifConId[-1])
            ifDst.append(ID)
            edfgEdgSrc.append(ifConId[-1])
            edfgEdgDst.append(ID)
            lastId = ID

    else:
        edfgEdgSrc.append(lastId)
        edfgEdgDst.append(ID)
        lastId = ID



    if len(children)==0:
        ast_path.append(copy.deepcopy(ast_path_single))

    ID += 1
    for child in children:
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

import pandas as pd
if  __name__ == '__main__':
    gcj_path = '../data/gcj_funcs.pkl'
    data = pd.read_pickle(gcj_path)
    print(data)
    temp=0
    for i,j in data.iterrows():
        print(j['code'])
        code = j["code"]
        temp+=1
        if temp==1:
            break
    #code = 'public class test{int a=4;string s = "a"; public void f1(){int b=0;b=a;s="op"; if(a>b && s!=""){a=-1;if(b==1) a=-3;}else{a=1;}} }'
    java_trans_path_list(code)
    print('edfgEdgSrc',edfgEdgSrc)
    print('edfgEdgDst',edfgEdgDst)
    print('ifSrc',ifSrc,'ifDst',ifDst)
    print('dataSrc',dataSrc,'dataDst',dataDst)



