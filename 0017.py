a = int(input("请输入一个数"))
a1 = a
b = 0
c = 0
while a1 != 0:
    c = int(a1) % 10
    a1 = int(a1)/10
    if c != 0:
        print(c,end = " ")
    else:
        pass
print(" ")

