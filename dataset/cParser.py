from pycparser import c_parser
from Node import ASTNode,SingleNode
import copy
import re
'''
实现：
1.节点替换
2.建立SG
3.edfg：分支结构和数据流动
'''
ID=0
depth = 0
varDec = False
memRef = False
forPar = False
varName2info = {}#由于解析工具的不同，他的深度应该保存的是Decl节点的深度
nodeInfo = {}
nodeInfoRep ={}
edfgEdgSrc = []
edfgEdgDst = []
edfgEdgInfo = []

sgEdgSrc = []
sgEdgDst = []
sgEdgInfo=[]
lastId = -1


'''
额外的测试变量
'''
declFlag = False
allNodeType = []
needRepNodeInfo = {}

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


def flagInit():
    global depth,varDec,memRef,forPar,varName2info,nodeInfo,nodeInfoRep
    global edfgEdgSrc,edfgEdgDst,edfgEdgInfo,sgEdgSrc,sgEdgDst,sgEdgInfo,lastId
    global declFlag,allNodeType,needRepNodeInfo,ifFlag,ifCon,ifConId,ifDepth,ifSrc,ifDst,dataSrc,dataDst,nodeDepth
    depth = 0
    varDec = False
    memRef = False
    forPar = False
    varName2info = {}  # 由于解析工具的不同，他的深度应该保存的是Decl节点的深度
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

    '''
    额外的测试变量
    '''
    declFlag = False
    allNodeType = []
    needRepNodeInfo = {}

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
        if k > id:
            break
    #c语言的解析中，节点的定义不会多也不会少，但是，使用中的节点可能不存在他的定义，因此存在查找不到的情况
    if tempId==-1:
        return
    tempType = varName2info[tempId][1]
    # print(memRef + str(id) + " type is " + tempType + " according to " + str(tempId))
    nodeInfo[id] = tempType
    edfgEdgSrc.append(tempId)
    edfgEdgInfo.append('value')
    edfgEdgDst.append(id)

#pycparser中所有的固有节点：
def read_cnode(nodePath='dataset/CnodeType.txt'):
    f = open(nodePath, encoding="utf-8")
    res = f.readlines()
    for type in res:
        allNodeType.append(type.strip("\n"))
    f.close()

def dataEdge(node):
    '''
    :param node: 传入一个节点
    :return: 返回从该节点出发第一个孩子下层次有多深
    '''
    if len(node.children()) == 0:
        return 1
    else:
        for x, child in node.children():
           return  dataEdge(child)+1


def node_dfs(node,digNum=0):
    global varName2info
    tempCurrent = SingleNode(node)
    tempToken = tempCurrent.get_token()
    if len(node.children())==0:
        return tempToken,digNum
    for x, y in node.children():
        return node_dfs(y,digNum+1)




def create_graph(node, depth=0):
    global ID,nodeInfo,varName2info,lastId,ifFlag,ifDepth,ifCon,ifConId,dataDst,dataSrc,batchDepth,nodeDepth
    parentID = ID
    current = SingleNode(node)
    token = current.get_token()
    nodeInfo[ID] = token
    nodeDepth.append(depth)
    '''
    如果token不在原定解析出节点里面，那么就放在待置换的字典中最后进行置换
    '''
    if token not in allNodeType and bool(re.match("^[A-Za-z0-9_]*$", token)):
        needRepNodeInfo[ID]=[token,ID,depth]
    '''
    变量类型和变量名之间的替换
    '''
    if token == 'Decl':
        varType,digNum = node_dfs(node)
        #即便在同一个路径会出现两个dec1在出现第二次的时候K都是ID，确定了出现的唯一性，对varName2info中的信息进行覆盖
        varName2info[ID+digNum-1]=[varType,depth]


    '''
    解析过程中Edfg的分支以及数据数据流动工作
    '''
    if depth <= ifDepth[-1]:
        # print('ifDepth[-1]',ID,ifDepth,ifConId,ifFlag,ifCon)
        # print('ifEdg',ifSrc,ifDst)
        ifFlag.pop(-1)
        ifCon.pop(-1)
        ifDepth.pop(-1)
        ifConId.pop(-1)
    if token == '=':
        # 直接通过递归遍历创建边
        count = 0
        for x,child in node.children():
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
            count += 1

    parentType = token
    if depth == ifDepth[-1] + 1 and ifFlag[-1][0]:  # 确定是ifstatement下的一个节点
        if ID != ifFlag[-1][1]+1:  # 如果是个分支节点要么是compoun或者是return ,if后面可能直接跟return,'<<‘if下直接跟cout
            if not ifCon[-1]:
                ifCon[-1] = True
                ifConId.append(lastId)
            ifSrc.append(ifConId[-1])
            ifDst.append(ID)
            edfgEdgSrc.append(ifConId[-1])
            edfgEdgDst.append(ID)
            edfgEdgInfo.append('route')
            lastId = ID
        else:#如果是个条件语句
            edfgEdgSrc.append(lastId)
            edfgEdgDst.append(ID)
            edfgEdgInfo.append('route')
            lastId = ID

    else:
        edfgEdgSrc.append(lastId)
        edfgEdgDst.append(ID)
        edfgEdgInfo.append('route')
        lastId = ID
    if token == 'If':#顺序问题，如果if恰好是另外一个if的执行语句则先处理上一个if，在对其本身信息进行存储
        ifFlag.append([True,ID])
        ifDepth.append(depth)
        ifCon.append(False)

    ID += 1
    if len(node.children()) != 0:
        for x, y in node.children():
            '''
            建立SG
            '''
            sgEdgSrc.append(parentID)
            sgEdgDst.append(ID)
            sgEdgInfo.append('parent')

            sgEdgSrc.append(ID)
            sgEdgDst.append(parentID)
            sgEdgInfo.append('child')

            create_graph(y, depth + 1)


def java_trans_path(node, ast_path_single, ast_path, depth=0):
    global ID
    current = SingleNode(node)
    token = current.get_token()
    ast_path_single.append(str(ID) + "  " + token + "-" + str(depth))
    ID += 1
    if len(node.children()) != 0:
        for x, y in node.children():
            java_trans_path(y, ast_path_single, ast_path, depth + 1)
            ast_path_single.pop()
    else:
        ast_path.append(copy.deepcopy(ast_path_single))


def java_trans_path_list(code):
    parser = c_parser.CParser()
    ast = parser.parse(code)
    ast_path_single=[]
    ast_path=[]
    java_trans_path(ast, ast_path_single,ast_path)
    for i in ast_path:
        print(i)
    return ast_path

def var_rep():
    '''
       变量的替换最终都转换成在nodeinfo里面的替换,下面都是节点替换的代码
       '''
    # 存储变量定义的信息
    for k, v in varName2info.items():
        if k not in nodeInfo:
            print('我竟然不存在', k)
        else:
            varName2info[k].insert(0, nodeInfo[k])
    # 存储变量使用时的信息
    # print('needRepNodeInfo', needRepNodeInfo)
    # 将两者中信息进行查找，并在最后的nodeInfo中进行更改
    for k, v in needRepNodeInfo.items():
        searchDec(v[0], v[1], v[2])

def list2txt(path,infoList,keyword):
    if type(infoList).__name__== 'dict':

        # print('write',infoList)
        with open(path, 'w') as f:
            f.write(keyword+'\n')
            for k,v in infoList.items():
                f.write(infoList[k]+' ')
            f.write('\n')
    else:
        if type(infoList).__name__ == 'list':
            with open(path, 'w') as f:
                f.write(keyword + '\n')
                for v in infoList:
                    f.write(str(v) + ' ')
                f.write('\n')

def print_code():
    # print('varName2info', varName2info)
    print('nodeInfo', nodeInfo)
    '''
    SG图是否构建成功测试
    '''
    print('sg\n',len(sgEdgSrc),sgEdgSrc)
    print(len(sgEdgDst),sgEdgDst)
    print(len(sgEdgInfo),sgEdgInfo)
    print('ifSrc', ifSrc)
    print('ifDst', ifDst)
    print('-' * 20, '\n', edfgEdgSrc)
    print(edfgEdgDst)
    print('dataSrc', dataSrc)
    print('dataDst', dataDst)


def parserC(code):
    parser = c_parser.CParser()
    ast = parser.parse(code)
    global  ID,allNodeInfo,allSgSrc,allSgDst,allEdfgSrc,allEdfgDst,allEdfgInfo,allSgInfo,batchAstNode,batchDepth
    tempId = ID
    # java_trans_path_list(code)
    ID = tempId
    # print(ID,'---',tempId)
    create_graph(ast)
    '''
    保存好数据之后的数据处理工作
    '''
    var_rep()
    edfgEdgSrc.pop(0)
    edfgEdgInfo.pop(0)
    edfgEdgDst.pop(0)

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
    # print('nodeInfo: ', len(nodeInfo), nodeInfo)
    # print('dataSrc: ', dataSrc)
    # print('dataDst: ', dataDst)
    # print('ifSrc: ', ifSrc)
    # print('ifDst: ', ifDst)
    # print('allnode', len(allNodeInfo), allNodeInfo)
    # print('sgInfo', sgEdgSrc, '\n', sgEdgDst)
    # print('edfgInfo', edfgEdgSrc, '\n', edfgEdgDst)
    '''
    只要不重置，都是顺序存储的，但是涉及到两个AST中相同变量类型和变量名之间的查找替换，以及edfg的源节点和目标节点需要剔除第一个
    '''
    flagInit()

def parserCForCorpus(code):

    flagInit()
    read_cnode()
    parser = c_parser.CParser()
    ast = parser.parse(code)
    global  ID,allNodeInfo,allSgSrc,allSgDst,allEdfgSrc,allEdfgDst,allEdfgInfo,allSgInfo,nodeInfo
    tempId = ID
    # java_trans_path_list(code)
    ID = tempId
    # print(ID,'---',tempId)
    create_graph(ast)
    '''
    保存好数据之后的数据处理工作
    '''
    var_rep()
    edfgEdgSrc.pop(0)
    edfgEdgDst.pop(0)
    edfgEdgInfo.pop(0)

    '''
    处理完一颗AST，对其信息进行保存
    '''
    return nodeInfo



def claPipline(data,dataSetName=''):
    '''
    每次处理一个批次的数据，同一批次的数据ID从0开始
    :param data:
    :return:
    '''
    global ID, batchAstNode, allNodeInfo, allSgSrc, allSgDst, allEdfgSrc, allEdfgDst, allSgInfo, allEdfgInfo,batchDepth
    ID = 0
    allNodeInfo = {}
    allSgSrc = []
    allSgDst = []
    allSgInfo = []
    allEdfgSrc = []
    allEdfgDst = []
    allEdfgInfo = []
    batchAstNode = []
    batchDepth = []


    if type(data).__name__ == 'tuple':
        for code in data:
            read_cnode()
            parserC(code)
    else:
        read_cnode()
        parserC(data)
    return allNodeInfo,allSgSrc,allEdfgSrc,allSgDst,allEdfgDst,batchAstNode,allSgInfo,allEdfgInfo,batchDepth

def cloPipline(data1,data2,dataSetName=''):
    '''
    每次处理一个批次的数据，同一批次的数据ID从0开始
    :param data:
    :return:
    '''
    global ID, batchAstNode, allNodeInfo, allSgSrc, allSgDst, allEdfgSrc, allEdfgDst, allSgInfo, allEdfgInfo,batchDepth
    ID = 0
    allNodeInfo = {}
    allSgSrc = []
    allSgDst = []
    allSgInfo = []
    allEdfgSrc = []
    allEdfgDst = []
    allEdfgInfo = []
    batchAstNode = []
    batchDepth = []

    srcSg1, srcEdfg1, dstSg1, dstEdfg1, nodeNum1, \
    srcSg2, srcEdfg2, dstSg2, dstEdfg2, \
    nodeNum2, sgInfo1, edfgInfo1, sgInfo2, edfgInfo2, nodeInfo1, nodeInfo2 ,depth1,depth2= [list() for x in range(18)]
    read_cnode()
    if type(data1).__name__ == 'tuple':
        for code in data1:
            parserC(code)
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
            parserC(code)

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
        print('cparser 482   clo c true true.....')
    return nodeInfo1, nodeInfo2, srcSg1, srcEdfg1, dstSg1, dstEdfg1, nodeNum1, \
           srcSg2, srcEdfg2, dstSg2, dstEdfg2, \
           nodeNum2, sgInfo1, edfgInfo1, sgInfo2, edfgInfo2,\
           depth1,depth2




if __name__ == '__main__':
    sava_path=['../data/parserCode/oj_node.txt','../data/parserCode/oj_sg.txt','../data/parserCode/oj_edfg.txt']
    import pandas as pd

    '''
    1.准备待解析的代码
    2.读取pycparser中固有的节点类型
    3.找出解析出的节点中所有的变量定义的信息
    4.找出解析出的节点中所有的变量使用（非定义)的节点
    '''

    read_cnode()
    # code = 'int test(int a){int firstNumber=3, secondNumber, sumOfTwoNumbers; if(secondNumber){ secondNumber=1;firstNumber=7;}else{ secondNumber=2;} ;printf("输入两个数(以空格分割): ");scanf("%d %d", &firstNumber, &secondNumber);sumOfTwoNumbers = firstNumber;sumOfTwoNumbers = firstNumber + secondNumber;printf("%d + %d = %d", firstNumber, secondNumber, sumOfTwoNumbers); return 0;}'
    # parserC(sava_path,code)


    code = 'int main(){int n,i,s;cin>>n; s=0;for (i=1;i<=n;i++)if (i % 7 !=0)if (i % 10 !=7)if (i/ 10 !=7){s=s+i*i;}cout<<s;} '
    # parserC(sava_path,code)
    parserCForCorpus(code)

    # print('main',allNodeInfo)
    # list2txt(sava_path[0], allNodeInfo, 'node')
    # list2txt(sava_path[1], allSgSrc, 'sgSrc')
    # list2txt(sava_path[1], allSgDst, 'sgDst')
    # list2txt(sava_path[2], allEdfgSrc, 'edfgSrc')
    # list2txt(sava_path[2], allEdfgDst, 'edfgDst')
    # print_code()







