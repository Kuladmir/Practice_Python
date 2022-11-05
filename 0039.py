a = {"intervention":"干涉","retain":"挽留"}
A = {"interference","retailer"}
B = {"干涉","零售商"}
#添加字典元素
b = dict(zip(A,B))
print(b)
#添加字典元素
a["interact"]="相互影响"
print(a)
#遍历
print(a["retain"],end=" ")
print(a.get("retention","No"))
#删除
del b["retailer"]
print(b)
print(b.pop("interference"))
#查找
print(a.get("Kuladmir"))
print(a.keys())
print(a.values())