a = int(input("请输入操作数:>"))
for i in range(1,a+1):
    for j in range(1,i+1):
        print("%d * %d == %2d"%(i,j,i*j),end="\t")
    print("")
