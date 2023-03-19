from tkinter import *
# from PIL import Image
win = Tk()
win.config(bg="#8Db6cd")
win.title("小平台")
win.geometry("500x500")
win.iconbitmap("./image/KuGouMusic.ico")
txt="Hello World"
msg=Message(win,text=txt,width=60,font=("宋体",20,"bold italic"))
msg.pack()
win.mainloop()