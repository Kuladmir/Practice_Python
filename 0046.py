sum=lambda x,y:x+y
a=int(input("请输入一个数"))
b=int(input("请再输入一个数"))
c=sum(a,b)
print(c)
# 匿名函数
def lam(d,e,fx):
    f=fx(d,e)
    print(f)
lam(5,10,lambda x,y:x*y)
lam(5,10,lambda x,y:x+y)
#匿名函数高级用法
