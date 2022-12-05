name_beg = '6602020'
list1 = []
for i in range(1 , 11):
    name_end = '%.3d'%(i)
    num = name_beg + name_end
    list1.append(num)
list2 = {}
for i in list1:
    list2[i] = 'python'
print('计算机编号  登录密码')
for key,value in list2.items():
    print('%s\t\t%s'%(key,value))


