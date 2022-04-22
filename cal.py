a=98.7
b=94.2
def cal_f1(x,y):
    f1 = (2 * x*y)/(x+y)
    return  f1


def all_(a,b,c,d,e):
    af=0.00455*a+0.00058*b+0.00243*c+0.01014*d+0.9823*e
    return af

def approch_cal(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5):
    p1 = cal_f1(x1,y1)
    p2 = cal_f1(x2,y2)
    p3 = cal_f1(x3,y3)
    p4 = cal_f1(x4,y4)
    p5 = cal_f1(x5,y5)

    p=all_(x1, x2, x3, x4, x5)
    r=all_(y1, y2, y3, y4, y5)

    print("T1  %.1f" % p1 )
    print("T2  %.1f" % p2)
    print("ST3 %.1f" % p3)
    print("MT3 %.1f" % p4)
    print("T-4 %.1f" % p5)
    print("P   %.1f" % p)
    print("R   %.1f" % r)
    print("f1  %.1f"%cal_f1(p,r))

print('********************OJ*********************')
print('astnn   %.1f'%cal_f1(98.9,92.7))
print('HGCR-   %.1f'%cal_f1(98.9,93.6))
print('HGCR    %.1f'%cal_f1(99.2,94.3))
print('********************GCJ*********************')
print('astnn   %.1f'%cal_f1(95.4,87.2))
print('FA-AST   %.1f'%cal_f1(96.3,85.5))
print('FCCA    %.1f'%cal_f1(96.7,89.8))
print('HGCR-   %.1f'%cal_f1(98.1,93.5))
print('HGCR    %.1f'%cal_f1(98.7,94.2))
print('********************BCB*********************')
print('--------ASTNN--------')
approch_cal(100,100,100,100,99.9,94.2,99.5,91.7,99.8,88.3)
print('--------FA-AST--------')
approch_cal(100,100,100,100,100,99.6,98.7,96.5,97.7,90.5)
print('--------FCCA--------')
approch_cal(100,100,100,100,100,99.8,98.7,95.9,98.2,92.3)
print('--------HGCR---------')
approch_cal(100,100,100,100,100,100,99.9,97.2,99.5,95.6)
