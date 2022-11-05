i = 1
while i==1:
    a = input("请输入用户名")
    b = input("请输入密码")
    lena = len(a)
    lenb = len(b)
    if 5<=lena<=15 and 6<=lenb<=20:
        print("用户名是%s,密码是%s"%(a,b))
        break
    else:
        print("用户名或密码不符合")




