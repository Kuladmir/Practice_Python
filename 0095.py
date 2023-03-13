from tkinter import *
win=Tk()
win.title("Linux")
win.geometry("500x200")
win.iconbitmap("./image/format.ico")
#创建滚动条
s=Scrollbar(win)
s.pack(side=RIGHT,fill=Y)
lb=Listbox(win,selectmode=MULTIPLE,height=5,yscrollcommand=s.set)
for i,item in enumerate(range(1,20)):
    lb.insert(i,item)
lb.pack()
#设置滚动条，使其垂直方向与Listbox组件内容绑定
s.config(command=lb.yview)
btn=Button(win,text="删除",command=lambda x=lb:x.delete(ACTIVE))
btn.pack(side="bottom")
win.mainloop()