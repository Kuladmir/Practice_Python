import tkinter as tk
def login():
    windows = tk.Tk()
    windows.title("C程序登录成功页面")
    windows.config(background="#FCABFC")
    win.iconbitmap("./image/KuGouMusic.ico")
    width = 400
    height = 400
    size = "%dx%d+%d+%d" % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    windows.geometry(size)
    text = tk.Label(windows,text="恭喜登录成功",bg="#73FFFF",fg="#7DCE2C",font=("宋体",25,"bold italic"))
    text.place(x=100,y=150)
    windows.mainloop()

win = tk.Tk()
win.title("C程序登录平台")
win.iconbitmap("./image/KuGouMusic.ico")
win.config(background="#FCABFC")
width = 400
height = 400
screenwidth = win.winfo_screenwidth()
screenheight = win.winfo_screenheight()
size = "%dx%d+%d+%d"%(width,height,(screenwidth-width)/2,(screenheight-height)/2)
win.geometry(size)
text = tk.Label(win,text="C程序用户登陆页面",font=("宋体",20,"bold"))
text.pack(side="top")
text1 = tk.Label(win,text="账号:",fg="green",font=("宋体",20,"italic"))
text1.place(x=50,y=100)
ent = tk.Entry(win,width=15,font=("宋体",20,"italic"))
ent.place(x=150,y=100)
text2 = tk.Label(win,text="密码:",fg="green",font=("宋体",20,"italic"))
text2.place(x=50,y=180)
ent = tk.Entry(win,width=15,font=("宋体",20,"italic"))
ent.place(x=150,y=180)
bu = tk.Button(win,text="登录",width=7,height=2,command=login)
bu.place(x=120,y=280)
bu1 = tk.Button(win,text="退出",width=7,height=2,command=quit)
bu1.place(x=240,y=280)
win.mainloop()
