a = ["obsess","aristocratic","offend"]
b = ["使迷恋","贵族","冒犯"]
c = dict(zip(a,b))
print(c)
c = dict(calculate="计算")
print(c)
# print(c.get("obsess","Error"))
print(c.keys())
print(c.values())
c.update({"hello":"world"})
print(c)