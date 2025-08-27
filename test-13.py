f = open("test-3.csv", "r+t", encoding="utf-8")
s = []
for i in f: # 实际得到的是一个字符串
    str = i.strip('\n')
    list = str.split(',')
    s.append(list)
print(s)
f.close()
