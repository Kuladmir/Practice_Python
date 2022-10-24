b = [1901,1951,1964,1954,1965]
a = ["Kula"]
c = ["Kuladmir","Apex","Legend",2019,3306,951]
#索引
print(c[2],c[-2])#Legend 3306
#切片
print(c[1:4])#Apex Legend 2019 3306
#乘法
print(a * 5)#Kula Kula Kula Kula Kula
#序列相加
print(a + b)
#检查成员
print("Apex"in c)
print("Kulapati" in c)
#序列长度
print(len(b),max(b),min(b))