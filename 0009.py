#判断一个年份是否为闰年
a = int(input("请输入一个年份:>"))
if a % 4 == 0 and a % 100 !=0:
    print("Yes")
else:
    if a % 400 == 0:
        print("Yes,century year")
    else:
        print("No")



