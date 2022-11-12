def prin(*args):
    print("i want to:",end=' ')
    for i in args:
        print(i,end=" ")
    print("")
prin("CAU","TsingHua")
a=["CAU","TsingHua","Peking"]
prin(*a)
def pri(**kwargs):
    for key,value in kwargs.items():
        print(key+"简称为"+value)
pri(北京='京',天津='津')
b={"天津":"津","云南":"云"}
pri(**b)


