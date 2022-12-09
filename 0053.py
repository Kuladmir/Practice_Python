class Custom(object):
    def __init__(self,name,money):
        self.name = name
        self.__money = money #定义私有属性
    def get_money(self):
        return self.__money
    def set_money(self,money):
        self.__money = money
cobj = Custom("Tom",100)
print(cobj.name)
cobj.name = "Harry"
cobj.set_money(100)
print(cobj.set_money(100))
