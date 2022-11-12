def prin(name,prize):
    '''
    :param name: 姓名
    :param prize: 奖励
    :return: NONE
    '''
    print('{}获得了{}'.format(name,prize))
prin("Kuladmir","1st")
# 位置参数
prin(prize="2nd",name="Yado")
# 关键字参数
def add(a,b=10):
    return a+b
print(add(5))
print(add(10,15))
# 默认参数

