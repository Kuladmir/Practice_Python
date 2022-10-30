py = ("Kuladmir","Admirmoon","Kulapati")
print(py[1])
print(py[:2])
for i in [0,1,2]:
    print(py[i],end=" ")
print("")
print(py.index("Kuladmir"))
pr = (("Apple",5),("Banana",4),("Lemon",2))
for i,ir in enumerate(pr):
    print("%.3d   %s   %.2f"%(i,ir[0],ir[1]))