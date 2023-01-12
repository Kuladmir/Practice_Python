import math
a = int(input("请输入一个数"))
for i in range(2,int(math.sqrt(a))+2,1):
    print(i)
    if(a % i == 0):
        print("%d不是素数"%a)
        break
if(i>int(math.sqrt(a))):
    print("%d是素数"%a)

