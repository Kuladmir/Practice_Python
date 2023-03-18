from tkinter import *
win = Tk()
win.title("页面介绍")
win.geometry("400x200+500+300")
win.iconbitmap("./image/format.ico")
text = Label(win,text="Kuladmir",bg="yellow",fg="red",font=("Times",24,"bold italic"))
text.place(x=140,y=50)
win.mainloop()