import pandas as pd
from tqdm import tqdm
from pycparser import c_parser, c_ast
import os
from dataset.Node import ASTNode, SingleNode
from javaNode import get_java_token,get_java_children



ojifNum,ojswitchNum=0,0
bcbifNum,bcbswitchNum=0,0
gcjifNum,gcjswitchNum=0,0
bcb_path='data/bcb_funcs.pkl'
gcj_path='data/gcj_funcs.pkl'
oj_path='data/ojclone.pkl'


def get_c_branch(node,ifNum,switchNum,sgNum,ndfgNum):
    global ojifNum,ojswitchNum
    current = SingleNode(node)
    sgNum[0]+=1
    ndfgNum[0]+=1

    if node.__class__.__name__=='ID':
        ndfgNum[0]+=1

    if current.get_token()=='If' :
        ifNum[0]+=1
    if current.get_token()=='Switch' :
        switchNum[0]+=1
    for _, child in node.children():
        get_c_branch(child,ifNum,switchNum,sgNum,ndfgNum)


def get_java_branch(node,ifNum,switchNum,sgNum,ndfgNum):
    token, children = get_java_token(node), get_java_children(node)
    sgNum[0]+=1
    ndfgNum[0]+=1
    if token == 'MemberReference':
        ndfgNum[0]+=1
    if token=='SwitchStatement':
        switchNum[0]+=1
    if token=='IfStatement':
        ifNum[0]+=1

    for child in children:
        get_java_branch(child,ifNum,switchNum,sgNum,ndfgNum)



def read_code(path):
    line_max_num,line_avg_num=0,0.0

    def get_line_num(str):
        return str.count("\n")

    data=pd.read_pickle(path)
    for i , j in tqdm(data.iterrows()):
        line_num=get_line_num(j['code'])
        line_max_num=line_num if line_num>line_max_num else line_max_num
        line_avg_num+=line_num

    return line_max_num,line_avg_num/len(data)
'''
---------------------------------------
数据统计调用
'''

def get_bcb_branch_start():
    source = pd.read_pickle(bcb_path)
    import javalang
    def parse_program(func):
        tokens = javalang.tokenizer.tokenize(func)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        return tree
    source['code']=source['code'].apply(parse_program)
    ifNum, switchNum = [0], [0]
    sgNumSum,ndfgNumSum,sgNumMax,ndfgNumMax,sgNumAvg,ndfgNumAvg=0,0,0,0,0.0,0.0
    file_num=len(source)
    pbar = tqdm(source.iterrows())
    for i, j in pbar:
        pbar.set_description("BCB Processing %d" % i)
        sgNum, ndfgNum = [0], [0]

        get_java_branch(j['code'],ifNum,switchNum,sgNum,ndfgNum)

        sgNumMax = sgNum[0] if sgNum[0] > sgNumMax else sgNumMax
        ndfgNumMax = ndfgNum[0] if ndfgNum[0] > ndfgNumMax else ndfgNumMax
        sgNumSum+=sgNum[0]
        ndfgNumSum+=ndfgNum[0]
    sgNumAvg=sgNumSum/file_num
    ndfgNumAvg=ndfgNumSum/file_num
    print('bcb\n', 'ifNum', ifNum, 'switchNum', switchNum,'sgNumMax',sgNumMax,'ndfgNumMax',ndfgNumMax,'sgNumAvg',sgNumAvg,'ndfgNumAvg',ndfgNumAvg)

def get_gcj_branch_start():
    source = pd.read_pickle(gcj_path)
    import javalang
    def parse_program(func):
        tokens = javalang.tokenizer.tokenize(func)
        parser = javalang.parser.Parser(tokens)
        tree=parser.parse()
        return tree
    source['code']=source['code'].apply(parse_program)
    ifNum, switchNum = [0], [0]
    sgNumSum, ndfgNumSum, sgNumMax, ndfgNumMax, sgNumAvg, ndfgNumAvg = 0, 0, 0, 0, 0.0, 0.0
    file_num = len(source)
    pbar=tqdm(source.iterrows())
    for i, j in pbar:
        pbar.set_description("GCJ Processing %d" % i)
        sgNum, ndfgNum = [0], [0]

        get_java_branch(j['code'], ifNum, switchNum, sgNum, ndfgNum)

        sgNumMax = sgNum[0] if sgNum[0] > sgNumMax else sgNumMax
        ndfgNumMax = ndfgNum[0] if ndfgNum[0] > ndfgNumMax else ndfgNumMax
        sgNumSum += sgNum[0]
        ndfgNumSum += ndfgNum[0]
    sgNumAvg = sgNumSum / file_num
    ndfgNumAvg = ndfgNumSum / file_num
    print('gcj\n', 'ifNum', ifNum, 'switchNum', switchNum, 'sgNumMax', sgNumMax, 'ndfgNumMax', ndfgNumMax, 'sgNumAvg',
          sgNumAvg, 'ndfgNumAvg', ndfgNumAvg)




def get_oj_branch_start():
    parser = c_parser.CParser()
    source=pd.read_pickle(oj_path)
    source['code'] = source['code'].apply(parser.parse)
    ifNum, switchNum = [0], [0]
    sgNumSum, ndfgNumSum, sgNumMax, ndfgNumMax, sgNumAvg, ndfgNumAvg = 0, 0, 0, 0, 0.0, 0.0
    file_num = len(source)
    pbar = tqdm(source.iterrows())
    for i, j in pbar:
        pbar.set_description("OJ Processing %d" % i)
        sgNum, ndfgNum = [0], [0]

        get_c_branch(j['code'], ifNum, switchNum, sgNum, ndfgNum)

        sgNumMax = sgNum[0] if sgNum[0] > sgNumMax else sgNumMax
        ndfgNumMax = ndfgNum[0] if ndfgNum[0] > ndfgNumMax else ndfgNumMax
        sgNumSum += sgNum[0]
        ndfgNumSum += ndfgNum[0]
    sgNumAvg = sgNumSum / file_num
    ndfgNumAvg = ndfgNumSum / file_num
    print('oj\n', 'ifNum', ifNum, 'switchNum', switchNum, 'sgNumMax', sgNumMax, 'ndfgNumMax', ndfgNumMax, 'sgNumAvg',
          sgNumAvg, 'ndfgNumAvg', ndfgNumAvg)

def line_inof_start():
    gcj_line_max, gcj_line_avg = read_code(gcj_path)
    print('gcj_line_max,gcj_line_avg\n', gcj_line_max, gcj_line_avg)
    bcb_line_max, bcb_line_avg = read_code(bcb_path)
    print('bcb_line_max,bcb_line_avg\n', bcb_line_max, bcb_line_avg)
    oj_line_max, oj_line_avg = read_code(oj_path)
    print('oj_line_max,oj_line_avg\n', oj_line_max, oj_line_avg)


# type_d={}
# source = pd.read_pickle('data/bcb_pair_ids.pkl')
# print(source)
# pbar = tqdm(source.iterrows())
# for i, j in pbar:
#     pbar.set_description("BCB Type Processing %d" % i)
#     if j['label'] in type_d:
#         type_d[j['label']] += 1
#     else:
#         type_d[j['label']] = 1
# sum_value=0
# for  value in type_d.values():
#     sum_value+=value
# for k, v in type_d.items():
#     print(k,v,v/sum_value)
# print(type_d)
oj_line_max, oj_line_avg = read_code(oj_path)
print('oj_line_max,oj_line_avg\n', oj_line_max, oj_line_avg)

# data=pd.read_pickle('data/ojclone.pkl')
# print(data)
# data.columns=['code']
# print(data)
# data.to_pickle('data/ojclone.pkl')
#
# # get_bcb_branch_start()
# # get_gcj_branch_start()
# get_oj_branch_start()
# id_pair=pd.read_pickle('data/oj_pair_ids.pkl')
# print(id_pair)
# iddict={}
# pbar = tqdm(id_pair.iterrows())
# func=pd.read_pickle('data/oj_funcs.pkl')
# print(func)
# for i, j in pbar:
#     pbar.set_description("BCB Type Processing %d" % i)
#     iddict[j['id1']]=func.loc[j['id1'],'code']
#     iddict[j['id2']] = func.loc[j['id2'], 'code']
# dataframe=pd.DataFrame.from_dict(iddict,orient='index')
# dataframe.to_pickle('data/ojclone.pkl')
