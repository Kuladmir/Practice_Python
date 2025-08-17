s1 = "--MuMu is a good game--"
print(s1.strip("-"))
print(s1.strip("/"))
s2 = "-/* MuMu is a good game--"
print(s2.strip("-/"))
print(s1.rjust(30,"*"))
print(s1.ljust(30,"*"))
print(s1.ljust(10))
print(s1.center(30,"*"))
print(s1.startswith("**"))
print(s1.endswith("--"))