i = 0
sec = 95645
while i < 3:
    a = int(input("请输入密码"))
    if a == sec:
        print("输入正确")
        break
    else:
        print("输入错误")
        i += 1
else:
    print("登入失败")