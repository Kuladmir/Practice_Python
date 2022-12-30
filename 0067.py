a = [50,20,303,5,99,46,501]
a.append(41)
print(a)
a.sort()
print(a)
b = sorted(a,reverse = True)
print(b)
print(a[1:5:1])
a.insert(1,60)
print(a)
b = ["Hello"]
print(b*5)
print(sum(a))
print(max(a))
for i,ir in enumerate(a):
    print(i,ir)