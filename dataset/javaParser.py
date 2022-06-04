import javalang
from javalang.ast import Node
import pandas as pd
from pycparser import c_parser
import copy
# from dataset.Node import ASTNode,SingleNode
import numpy as np

'''
使用的变量的解释
ID：每个节点的ID，从0开始
depth: 每个节点所在的层次，从0开始
nodeInfo:  存储所有node信息，key:ID, value:token
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


lastId = -1

#处理分支结构,可能出现嵌套的情况，因此类似括号匹配，用栈去做
ifFlag = []
ifCon = []
ifConId = [-1]
ifDepth = [-999]


#分支结构的边，以及局部数据流动的边
ifSrc = []
ifDst = []
dataSrc = []
dataDst = []

#将不同AST中的信息汇总起来
allNodeInfo={}
allSgSrc = []
allSgDst = []
allSgInfo = []
allEdfgSrc = []
allEdfgDst = []
allEdfgInfo = []
batchAstNode = []
batchDepth = []
nodeDepth = []

#克隆需要两份存储

def flagInit():
    global depth,varDec,memRef,forPar,varName2info,nodeInfo,nodeInfoRep,batchAstNode,nodeDepth
    global edfgEdgSrc,edfgEdgDst,edfgEdgInfo,sgEdgSrc,sgEdgDst,sgEdgInfo,lastId
    global declFlag,allNodeType,needRepNodeInfo,ifFlag,ifCon,\
        ifConId,ifDepth,ifSrc,ifDst,dataSrc,dataDst
    ID = 0
    depth = 0
    varDec = False
    memRef = False
    forPar = False
    varName2info = {}
    nodeInfo = {}
    nodeDepth = []
    nodeInfoRep = {}
    edfgEdgSrc = []
    edfgEdgDst = []
    edfgEdgInfo = []


    sgEdgSrc = []
    sgEdgDst = []
    sgEdgInfo = []

    lastId = -1

    # 处理分支结构,可能出现嵌套的情况，因此类似括号匹配，用栈去做
    ifFlag = []
    ifCon = []
    ifConId = [-1]
    ifDepth = [-999]

    # 分支结构的边，以及局部数据流动的边
    ifSrc = []
    ifDst = []
    dataSrc = []
    dataDst = []


def nodeName2Type():
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
    # print(tempId,memRef)
    # print('search', memRef, id,tempId)
    # print(varName2info)
    if tempId > 0:
        '''
        如果同时定义两个变量，则就是需要遍历已经存的nodeInfo
        547 ['b2', 'int', 6]
        549 ['b3', 'b2', 6]
        '''
        need=True
        tempType = varName2info[tempId][1]
        #就很奇葩，会有变量和他的类型一摸一样，这样的时候就不用去找替换了，标志位置为false
        if(varName2info[tempId][1]==varName2info[tempId][0]):
            need = False
        FORNUM =0
        while(need):
            FORNUM+=1
            # if id ==67:
            #     print('=='*80,tempType)
            #     print('varName2info',varName2info)
            # print('woyousixunhuanle ')
            replace = False
            for k,v in varName2info.items():
                if FORNUM==9:
                    need = False
                    break
                if v[1] != 'ReferenceType'and v[0] == tempType  :
                    replace = True
                    tempId = k
                    tempType = v[1]
            if replace == False:
                need = False

        # print(memRef + str(id) + " type is " + tempType + " according to " + str(tempId))
        nodeInfoRep[id] = tempType
        edfgEdgSrc.append(tempId)
        edfgEdgInfo.append('value')
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


def create_graph(node,depth=0,parentType=''):

    global ID,varDec, memRef,nodeInfo,varName2info,forPar,lastId,ifFlag,ifDepth,ifCon,ifConId,dataDst,dataSrc,nodeDepth,batchDepth
    parentID=ID
    token, children = java_get_token(node), java_get_children(node)
    nodeInfo[ID] = token
    nodeDepth.append(depth)
    '''
    解析过程中的变量类型替换变量名的的工作
    不能每次都变换parentType，因为VariableDeclarator下的两个节点，可能才出现定义的变量的名称
    '''
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
    '''
    解析过程中Edfg的分支以及数据数据流动工作
    '''
    if depth <= ifDepth[-1]:
        # print('='*80,'delete')
        # print(ID,token,)
        # print('ifCon',ifCon)
        # print('ifconID',ifConId)
        # print('ifDEp',ifDepth)
        ifFlag.pop(-1)
        ifCon.pop(-1)
        ifDepth.pop(-1)
        ifConId.pop(-1)

    if token == 'Assignment':
        #直接通过递归遍历创建边
        count = 0
        for child in children:
            if count == 0:
                dataDst.append(ID + dataEdge(child))
                edfgEdgDst.append(ID + dataEdge(child))
            else:
                if count == 1:
                    dataSrc.append(dataDst[-1] + dataEdge(child))
                    edfgEdgSrc.append(dataDst[-1] + dataEdge(child))
                    edfgEdgInfo.append('value')
                else:
                    break
            count+=1

    parentType = token
    if depth == ifDepth[-1]+1 and ifFlag[-1][0] :#确定是ifstatement下 的一个节点
        if ID == ifFlag[-1][1]+1:#条件语句.如果是个分支节点,如果带大括号他就是个BlockStatement如果没有大括号而是单个语句，他就是个StatementExpression,它还可能是通过计算得到的，还是通过
            edfgEdgSrc.append(lastId)
            edfgEdgDst.append(ID)
            edfgEdgInfo.append('route')
            lastId = ID
        else: #分支语句
            if not ifCon[-1]:
                ifCon[-1] = True
                ifConId.append(lastId)
            ifSrc.append(ifConId[-1])
            ifDst.append(ID)
            edfgEdgSrc.append(ifConId[-1])
            edfgEdgDst.append(ID)
            edfgEdgInfo.append('route')
            lastId = ID
    else:#如果不是ifstatment的分支节点呢，就直接继续往下执行，是按顺序的route节点
        edfgEdgSrc.append(lastId)
        edfgEdgDst.append(ID)
        edfgEdgInfo.append('route')
        lastId = ID
    if token == 'IfStatement':
        ifFlag.append([True,ID])
        ifDepth.append(depth)
        ifCon.append(False)
    ID+=1
    for child in children:
        childtoken = java_get_token(child)
        childtoken = childtoken.replace("\n", "")
        list = childtoken.split()
        if len(list)==0:
            continue

        '''
        建立SG的父子边
        '''
        sgEdgSrc.append(parentID)
        sgEdgDst.append(ID)
        sgEdgInfo.append('parent')

        sgEdgSrc.append(ID)
        sgEdgDst.append(parentID)
        sgEdgInfo.append('child')
        # if forPar:
        #     print(ID,child,type(child))
        if isinstance(child,str):
            if memRef:
                searchDec(child,ID,depth)
            if varDec:
                varName2info[ID] = [child,nodeInfo[ID-2],depth]
                '''
                添加完之后说明varDec已经被用过了，因为他可能解析出来很多VARDEC，但是，用过一次就不能重复用了，一个定义声明只定义一个变量
                '''
                varDec = False
                parentType='used'
            if forPar:
                varName2info[ID] = [child, nodeInfo[ID - 1], depth]
                parentType = 'used'
                varDec = False
        create_graph(child,depth+1,parentType)

def java_trans_path(node, ast_path_single, ast_path, depth=0):
    global ID
    token, children = java_get_token(node), java_get_children(node)
    # print('path  ======>',ID,token)

    ast_path_single.append(str(ID) + "  " + token + "-" + str(depth))
    ID += 1
    if len(children) == 0:
        ast_path.append(copy.deepcopy(ast_path_single))
    for child in children:
        java_trans_path(child, ast_path_single, ast_path, depth + 1)
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



def java_trans_path_list(code,dataSetName='bcb'):
    ast=get_java_ast(code,dataSetName)
    ast_path_single=[]
    ast_path=[]
    java_trans_path(ast, ast_path_single,ast_path)
    for i in ast_path:
        print(i)
    return ast_path

def parserJava(code,dataSetName = 'bcb'):
    ast=get_java_ast(code,dataSetName = dataSetName)
    global ID,allNodeInfo,allSgSrc,allSgDst,\
        allEdfgSrc,allEdfgDst,allSgInfo,\
        allEdfgInfo,batchAstNode,batchDepth,batchDepth
    tempId = ID
    # java_trans_path_list(code,dataSetName = dataSetName)
    ID = tempId
    # print(ID,'---',tempId,'=====================================>new code')
    create_graph(ast)
    '''
    保存好数据之后的数据处理工作
    '''
    nodeName2Type()
    edfgEdgSrc.pop(0)
    edfgEdgDst.pop(0)
    edfgEdgInfo.pop(0)


    '''
    处理完一颗AST，对其信息进行保存
    '''
    batchAstNode.append(len(nodeInfo))
    allNodeInfo.update(nodeInfo)
    allSgSrc += sgEdgSrc
    allSgDst += sgEdgDst
    allEdfgSrc += edfgEdgSrc
    allEdfgDst += edfgEdgDst
    allSgInfo += sgEdgInfo
    allEdfgInfo += edfgEdgInfo
    batchDepth += nodeDepth

    '''
    中间的边生成
    '''
    # print('nodeInfo: ',len(nodeInfo), nodeInfo)
    # print('dataSrc: ', dataSrc)
    # print('dataDst: ',dataDst)
    # print('ifSrc: ', ifSrc)
    # print('ifDst: ', ifDst)
    # print('allnode',len(allNodeInfo),allNodeInfo)
    # print('sgInfo',sgEdgSrc,'\n',sgEdgDst)
    # print('edfgInfo',edfgEdgSrc,'\n',edfgEdgDst)

    '''
    只要不重置，都是顺序存储的，但是涉及到两个AST中相同变量类型和变量名之间的查找替换，以及edfg的源节点和目标节点需要剔除第一个
    '''
    flagInit()

def parserJavaForCorpus(code,dataSetName = 'bcb'):
    flagInit()
    ast=get_java_ast(code,dataSetName = dataSetName)
    global ID,allNodeInfo,allSgSrc,allSgDst,allEdfgSrc,allEdfgDst,allSgInfo,allEdfgInfo,nodeInfo
    tempId = ID
    # java_trans_path_list(code,dataSetName = dataSetName)
    ID = tempId
    create_graph(ast)
    nodeName2Type()
    edfgEdgSrc.pop(0)
    edfgEdgDst.pop(0)
    # print('ifSrc: ', ifSrc)
    # print('ifDst: ', ifDst)
    return nodeInfo



def pipline(data1,data2,dataSetName = 'bcb'):

    #克隆需要把两份代码存储起来
    global ID, batchAstNode,allNodeInfo,allSgSrc,allSgDst,allEdfgSrc,allEdfgDst,allSgInfo,allEdfgInfo,batchDepth
    ID =0
    allNodeInfo = {}
    allSgSrc = []
    allSgDst = []
    allSgInfo = []
    allEdfgSrc = []
    allEdfgDst = []
    allEdfgInfo = []
    batchAstNode = []
    sava_path = ['../data/parserCode/' + dataSetName + '_node.txt', '../data/parserCode/' + dataSetName + '_sg.txt',
                 '../data/parserCode/' + dataSetName + '_edfg.txt']
    srcSg1,srcEdfg1,dstSg1,dstEdfg1,nodeNum1, \
    srcSg2, srcEdfg2, dstSg2, dstEdfg2,\
    nodeNum2,sgInfo1,edfgInfo1,sgInfo2,edfgInfo2,nodeInfo1,nodeInfo2,depth1,depth2= [list() for x in range(18)]
    if type(data1).__name__ == 'tuple':
        for code in data1:
            parserJava(code,dataSetName = dataSetName)
        srcSg1 = copy.deepcopy(allSgSrc)
        srcEdfg1 = copy.deepcopy(allEdfgSrc)
        dstSg1 = copy.deepcopy(allSgDst)
        dstEdfg1 = copy.deepcopy(allEdfgDst)
        nodeNum1 = copy.deepcopy(batchAstNode)
        sgInfo1 = copy.deepcopy(allSgInfo)
        edfgInfo1 = copy.deepcopy(allEdfgInfo)
        nodeInfo1 = copy.deepcopy(allNodeInfo)
        depth1 = copy.deepcopy(batchDepth)


    allNodeInfo = {}
    allSgSrc = []
    allSgDst = []
    allSgInfo = []
    allEdfgSrc = []
    allEdfgDst = []
    allEdfgInfo = []
    batchAstNode = []
    batchDepth = []
    ID = 0
    if type(data2).__name__ == 'tuple':
        for code in data2:
            parserJava(code,dataSetName = dataSetName)
        srcSg2 = copy.deepcopy(allSgSrc)
        srcEdfg2 = copy.deepcopy(allEdfgSrc)
        dstSg2 = copy.deepcopy(allSgDst)
        dstEdfg2 = copy.deepcopy(allEdfgDst)
        nodeNum2 = copy.deepcopy(batchAstNode)
        sgInfo2 = copy.deepcopy(allSgInfo)
        edfgInfo2 = copy.deepcopy(allEdfgInfo)
        nodeInfo2 = copy.deepcopy(allNodeInfo)
        depth2 = copy.deepcopy(batchDepth)
    if len(depth1)==len(nodeInfo1) and len(depth2)==len(nodeInfo2):
        print('java 488   clo c true true.....')
    return nodeInfo1,nodeInfo2,srcSg1,srcEdfg1,dstSg1,dstEdfg1,nodeNum1, \
    srcSg2, srcEdfg2, dstSg2, dstEdfg2,\
    nodeNum2,sgInfo1,edfgInfo1,sgInfo2,edfgInfo2,\
    depth1,depth2
# def gcjPipline(data,dataSetName=''):
#     '''
#     每次处理一个批次的数据，同一批次的数据ID从0开始
#     :param data:
#     :return:
#     '''
#     global ID, batchAstNode, allNodeInfo, allSgSrc, allSgDst, allEdfgSrc, allEdfgDst, allSgInfo, allEdfgInfo
#     ID = 0
#     allNodeInfo = {}
#     allSgSrc = []
#     allSgDst = []
#     allSgInfo = []
#     allEdfgSrc = []
#     allEdfgDst = []
#     allEdfgInfo = []
#     batchAstNode = []
#
#     if type(data).__name__ == 'tuple':
#         for code in data:
#             parserJava(code,dataSetName = dataSetName)
#     return allNodeInfo,allSgSrc,allEdfgSrc,allSgDst,allEdfgDst,batchAstNode,allSgInfo,allEdfgInfo


import pandas as pd
if  __name__ == '__main__':
    import pandas as pd

    dataPath = '../data/bcb_funcs.pkl'
    data = pd.read_pickle(dataPath)

    code = data.loc[48908, 'code']
    print(code)
    nodeInfo = parserJavaForCorpus(code, 'bcb')

    corpus = list(nodeInfo.values())
    corpus_str = ' '.join(c.replace("\n", "") for c in corpus)




    # code = 'public class test{int a=4;string s = "a"; public void f1(){int b=a;b=9;s="op"; if(true){System.out.println(a);}else{return b;}} }'
    # parserJavaForCorpus(code,'bcb')
    # code = 'public class test{int b=4;string sss = "a"; public void f1(){int b=a;b=9;sss="op"; if(true){System.out.println(a);}else{return b;}} }'
    # parserJavaForCorpus(code, 'bcb')
    # print("================")
    # print('-'*20,'输出信息查看','-'*20)
    # print(len(allEdfgInfo),len(allEdfgDst))
    # print('edfg:\n',allEdfgSrc)
    # print(allEdfgDst)
    # print(allEdfgInfo)
    # print('sg\n',allSgSrc)
    # print(allSgDst)
    # print(allSgInfo)
    # print('nodeInfo',allNodeInfo)
    #
    #
    #
