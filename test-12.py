f = open("test-3.csv", "w+", encoding="utf-8")
ss = [
    ["Name", "ID", "Sex"],
    ["Dmir", "0845", "M"],
    ["Kuda", "0889", "W"],
    ["Rela", "0545", "M"],
    ]
for i in ss:
    s = ','.join(i) + '\n'
    f.write(s)
f.seek(0)
read = f.readlines()
print(read)
f.close()