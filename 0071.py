class Inter:
    def __init__(self):
        self.wage = 0
    def ask(self):
        print("You can ask a qusetion")
    def __talk_wage(self):#不能直接调用
        print("Calculate Wage")
    def talk_wage(self):
        if self.wage > 2000:
            print("High")
        else:
            print("Ok")
Me = Inter()
Me.ask()
Me.wage = 20050
Me.talk_wage()