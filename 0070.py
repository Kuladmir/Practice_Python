class Custom:
    def __init__(self,name,money):
        self.name = name
        self.__money = money
    def get_money(self):
        return self.__money
    def set_money(self,money):
        self.__money = money
        return self.__money
cobj = Custom("Tom",100)
print(cobj.name)
cobj.name = "Harry"
print(cobj.get_money())
print(cobj.set_money(1000))