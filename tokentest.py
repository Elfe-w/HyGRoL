import javalang
from javalang.ast import Node
import pandas as pd
from pycparser import c_parser
import copy
from dataset.Node import ASTNode,SingleNode

ID=0
'''
使用的变量的解释
varName2info：  变量名：【id，层，变量类型】
nodeInfo:  存储所有node信息，key:nodetype, value:ID
'''
varName2info = {}

def checkDec(token,subToken):
    '''
    :param token: 节点的type methodeDeclaration
    :param subToken: 之后判断他是一个Declarator 还是memberReference
    :return:
    '''
    if token.find(subToken) :
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

varDec = False
memRef = False
def java_trans_path(node,ast_path_single, ast_path):
    global ID
    token, children = java_get_token(node), java_get_children(node)
    # print(token,'--------',children)

    varDec = False
    memRef = False


    if(checkDec(token,'VariableDeclarator')):
        varDec = True
    else:
        if(checkDec(token,"MemberReference")):
            memRef = True

    ast_path_single.append(str(ID)+token)
    ID+=1
    if len(children)==0:
        ast_path.append(copy.deepcopy(ast_path_single))

    for child in children:
        if isinstance(child,str):
            if memRef:
                print("menRef->child    "+ child)
            if varDec:
                print("varDec->child    " + child)
        java_trans_path(child, ast_path_single,ast_path)
        ast_path_single.pop()

def get_java_ast(code):
    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree


def java_trans_path_list(code):
    ast=get_java_ast(code)
    ast_path_single=[]
    ast_path=[]
    java_trans_path(ast, ast_path_single,ast_path)
    for i in ast_path:
        print(i)
    return ast_path

if  __name__ == '__main__':
    code = 'public class test{int a=4; public void f1(){int b=0;b=9; if(true){a=-1;}else{a=1;}} }'
    java_trans_path_list(code)

