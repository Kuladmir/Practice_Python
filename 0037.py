a = [1962,1970,1978]
a.append(1986)
print(a)
b = [1994,2002]
a.extend(b)
print(a)
a.insert(0,1954)
print(a)
c = [1938,1946]
a.insert(0,c)
print(a)
for i in a:
    print(i,end=' ')