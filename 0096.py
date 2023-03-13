from tkinter import *
from tkinter import messagebox
win = Tk()
win.title("获取列表框")
win.geometry("500x500+200+200")
#创建变量，用var接收鼠标点击的选项
var = StringVar()
lblinfo = Label(win,bg="#00ff00",font=("宋体",20,"bold italic"),width=20,textvariable=var)
lblinfo.pack()
def click_button():
    try:
        val = lb.get(lb.curselection())
        #设置Label的值
        var.set(val)
    except Exception as e:
        e = "发现错误"
        messagebox.showinfo(e,"发现错误")

#创建一个按钮并放置，单击按钮调用print_selectitem
bt = Button(win,text="获取选项",command=click_button)
bt.pack()
#创建Listbox，并添加内容
lbvar = StringVar()
lbvar.set(["C","Python","Java"])
#传概念Listbox，通过listvariable传递变量
lb = Listbox(win,listvariable=lbvar)
lb.pack()
win.mainloop()

