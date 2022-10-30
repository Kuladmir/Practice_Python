a = {'1':"Kula",'2':"mmd"}
a['3'] = "apple"
print(a)
print(a.pop('3'))
print(a)
print(a.get('1'),end=" ")#检查是否有元素键为1
print(a.get('3'))#检查是否有元素键为3
print(a.values())#输出字典的元素值
print(a.keys())#输出字典的元素键
