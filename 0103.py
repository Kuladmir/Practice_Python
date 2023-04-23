from tkinter import *
import tkinter.messagebox
win=Tk()
win.config(bg="#87CEEB")
win.title("记事本")
win.geometry("450x350+300+200")
win.iconbitmap("./image/format.ico")
def menuCommand():
    tkinter.messagebox.showinfo("主菜单栏","你正在使用")
#创建一个主目录
main_menu = Menu(win)
main_menu.add_command(label="文件",command=menuCommand)
main_menu.add_command(label="编辑",command=menuCommand)
main_menu.add_command(label="查看",command=menuCommand)
main_menu.add_command(label="工具",command=menuCommand)
main_menu.add_command(label="帮助",command=menuCommand)
win.config(menu=main_menu)
win.mainloop()