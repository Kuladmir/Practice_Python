a = ["Hello"]
b = ("Hello",)
c = {'1':'Hello'}
d = {1,5,6}
print(a*5)
print(b*5)
a1 = ["World"]
b1 = ("World",)
c1 = {'2':"World"}
d1 = {9,10,15}
print(a+a1)
print(b+b1)
# print(c+c1)
# print(d+d1)
print(5 in a)
print(5 in b)
print(5 in c)
print(5 in d)
for i in a:
    print(i)
for i in b:
    print(i)
for i in c:
    print(i)
for i in d:
    print(i)
print(a.index("Hello"))
print(b.index("Hello"))
# print(c.index("1"))
# print(d.index(5))
print(sum(d))

