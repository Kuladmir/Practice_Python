a = {'a':"Kula",'1':"dmir",'c':"Adi"}
b1 = ["cau","pking","tsing","2"]
b2 = ["中国农业大学","北京大学","清华大学","未知"]
b = dict(zip(b1,b2))
c = dict(ala="万国",bili=20,kud='a')
print(a["a"],end=" ")#指定输出
print(a["1"])#指定输出
print(b)#全部输出
print(b.get("cau","default:不存在"))#指定输出2
for item in c.items():
    print(item,end="")#全部遍历
print("")
a["5"]="Unknown"#添加新元素
print(a)
del a["5"]#删除元素
print(a,end=" ")
a.clear()#清空
print(a)

