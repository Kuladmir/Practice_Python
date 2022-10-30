a = "Kuladmir"
print(a.replace('u','p'),end='  ')#将u替换为p
b = "Assassin Creed"
print(b.replace('ss','aa',1))#将ss替换为aa
c="I have a pen"
print(c.split(),end='   ')
print(c.split('a'),end='   ')#以a为切割标志
print(c.split(' ',2))#切割两次
str = c.split()
print(str)
print('*'.join(str))
