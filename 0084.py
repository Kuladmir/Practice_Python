import tkinter as tk
from tkinter import messagebox
win = tk.Tk()
win.title("欢迎使用")
win.geometry("400x300+200+200")
win.iconbitmap("./image/KuGouMusic.ico")
win.resizable(0,0)

tk.Label(win,text="用户名").grid(row=0)
tk.Label(win,text="密码").grid(row=1)

e = tk.Entry(win)
e2 = tk.Entry(win,show="*")
e.grid(row=0,column=1,padx=10,pady=5)
e2.grid(row=1,column=1,padx=10,pady=5)

def login():
    messagebox.showinfo("欢迎来到这里")

button=tk.Button(win,text="登录",width=10,command=login)
button.grid(row=3,column=0,sticky="w",padx=10,pady=5)
button=tk.Button(win,text="退出",width=10,command=quit)
button.grid(row=3,column=1,sticky="e",padx=10,pady=5)
win.mainloop()
