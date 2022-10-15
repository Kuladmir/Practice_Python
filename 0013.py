print("请输入你的密码，注意，你只有五次机会")
password = 5
i = 1
while i <= 5:
    a = int(input("请输入:>"))
    if a == password:
        print("登陆成功")
        i = 5
    else:
        print("登录失败，以错误%d次"%i)
    i += 1

