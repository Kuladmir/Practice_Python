from tkinter import *
win = Tk()
win.title("List 测试")
win.geometry("500x400+200+200")
win.iconbitmap("./image/format.ico")
l1 = Listbox(win)
l1.pack()
# for i,item in enumerate(["C语言","Python","Java"]):
#     l1.insert(i,item)
for i in(["C语言","Python","Java"]):
    l1.insert("end",i)
win.mainloop()
