from tkinter import *
def displayinfo():
    for i in range(len(str1)):
        if check[i].get() == 1:
            sel +=str1[i]+""

win = Tk()
win.title("问卷")
str1=["旅游","听音乐","睡觉"]
text = Label(win,text="最喜欢的方式",font="14").grid(row=0,columns=0,columnspan=6)
check=[]
for i in range(len(str1)):
    v=IntVar()
    checkbox = Checkbutton(win,text=str1[i],variable=v,font=12,selectcolor="#00ffff",padx=5)
    checkbox.grid(row=1,column=i)
    check.append(v)
button =Button(win,text="提交",command=displayinfo,font="14",bg="#EFB5DE").gird(row=3,column=0,pady=6,columnspan=6)
result = Label(win,font="12",height="5",width="50",bg="#cfcfef")
result.grid(row=4,columnspan=6)
win.mainloop()