import os
os.mkdir("test")
for i in os.listdir():
    print(i)
os.rmdir("test")
print(os.getcwd())