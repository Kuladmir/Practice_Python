import tkinter as tk
import time
win = tk.Tk()
win.title("时间")
win.iconbitmap("./image/KuGouMusic.ico")
win.geometry("500x500+100+100")
win.resizable(0,0)

def gettime():
    dstr.set(time.strftime("%H:%M:%S"))
    win.after(1000,gettime)

dstr = tk.StringVar()
lbl = tk.Label(win,textvariable=dstr,fg="green",font=("黑体",20,"bold italic")).pack()
gettime()
win.mainloop()