a = [1,2,3]#列表操作
b = [5,6]
#添加元素
a.append(4)#[1,2,3,4]
a.extend(b)#[1,2,3,4,5,6]
a.insert(0,0)#[0,1,2,3,4,5,6]
print(a)
#删除元素
a.remove(2)#[0,1,3,4,5,6]
a.pop(2)#[0,1,4,5,6]
print(a)
#切片
print(a[1:3:1])#[1,4]
#遍历
for i in a:
    print(i,end=' ')#0 1 4 5 6
print("")
#相加、相乘
print(a+b)
print(b * 2)
#排序，求和
a.reverse()
print(a)#[6,5,4,1,0]
print(sum(b))#11
a.sort()
print(a)#[0,1,4,5,6]