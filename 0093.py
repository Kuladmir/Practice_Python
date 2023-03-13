from tkinter import *
win = Tk()
win.title("Text的使用")
win.geometry("500x600")
text = Text(win,width=35,height=35)
text.pack()
text.insert(INSERT,"welcome cau\n")
text.insert(INSERT,"计算中心")#继续向后插入文字
win.mainloop()