a = float(input("请输入总价格:>"))
b = float(input("请输入当前余额:>"))
if (a>=100):
    a*=0.9
if(b>a):
    print("支付成功，消费%.2f元，找回%.2f元"%(a,b-a))
else:
    print("支付失败，差%.2f元"%(a-b))

count = 0
for i in range(1,101,1):
    if(i%3 == 0):
        count+=i
print(count)