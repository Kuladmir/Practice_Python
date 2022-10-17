a = 0
i = 0
for i in range(1,10):
    a += i
else:
    print("完成了")
print(a)
print("--------")
a = 0
for i in range(1,10):
    a += i
    if i == 9:
        break
else:
    print("完成了")
print(a)