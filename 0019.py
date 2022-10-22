age = 24
i = 0
for i in [1,2,3]:
    b = int(input("请输入年龄"))
    if b == age:
        print("恭喜猜对了信息")
        break
    else:
        print("猜错了")
else:
    print("猜错了，再见")