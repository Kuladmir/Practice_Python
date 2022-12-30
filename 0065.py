a = int(input("输入一个年份"))
if ((a%4==0 and a%100!=0)or(a%400==0)):
    print("True")
else:
    print("False")

b = int(input("请输入秒数"))
c = int(b/3600)
d = int((b-3600*c)/60)
e = b-c*3600-d*60
print("%d 小时: %d 分钟: %d 秒"%(c,d,e))
