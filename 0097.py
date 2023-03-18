from tkinter import *
from tkinter.ttk import *
from tkinter.messagebox import *
import pymysql

def login():
    userName = EntryUser.get()
    userPwd = EntryPwd.get()
    if(userName=="" or userPwd==""):
        showerror("发生错误！","用户名或密码为空")
    else:
        db = pymysql.connect(host="localhost", user="root", password="123456", db="membership")
        cursor = db.cursor()
        strsql = "select * from db_login where loginuser='" + userName + "'and loginpwd='" + userPwd + "';"
        # print(strsql)
        try:
            cursor.execute(strsql)
            result = cursor.fetchall()
            print(result)
            if(len(result)>0 and CheckVar_1.get()==1):
                showinfo("提示","您以管理员身份登录成功")
            elif(len(result)>0 and CheckVar_1.get()==0):
                showinfo("提示","登录成功")
            else:
                if(len(result)==0):
                    showerror("错误","账号不存在")
                    EntryUser.delete(0,END)
                    EntryPwd.delete(0,END)
                else:
                    showerror("错误","账号或密码错误")
                    EntryPwd.delete(0,END)
        except:
            db.rollback()
            return 0
        finally:
            cursor.close()
            db.close()

def regist():
    userName = EntryUser.get()
    userPwd = EntryPwd.get()
    if (userName == "" or userPwd == ""):
        showerror("发生错误！", "用户名或密码为空")
    else:
        db = pymysql.connect(host="localhost", user="root", password="123456", db="membership")
        cursor = db.cursor()
        cursor.execute("select * from db_login where loginuser='"+userName+"';")
        result =cursor.fetchall()
        if(len(result)!=0):
            showerror("注意","该用户名已存在")
            EntryUser.delete(0, END)
            EntryPwd.delete(0,END)
        else:
            if(CheckVar_1.get()==0):
                strsql = "INSERT INTO db_login(loginuser,loginpwd) VALUES('"+userName+"','"+userPwd+"');"
                try:
                    cursor.execute(strsql)
                    db.commit()
                    result = cursor.fetchall()
                    showinfo("提示", "您已注册成功")
                except:
                    db.rollback()
                    return 0
                finally:
                    cursor.close()
                    db.close()
            else:
                showerror("错误", "请勿勾选管理员选项")
                cursor.close()
                db.close()

loginwin =Tk()
loginwin.title("会员管理系统")
loginwin.resizable(0,0)
loginwin.iconbitmap("./image/format.ico")
loginwin.geometry("400x320+550+200")
Label(loginwin,text="          欢迎登录",font=("宋体",20,"bold italic")).place(x=0,y=0,width=360,height=80)
Label(loginwin,text="用户名",font=("宋体",15,"bold")).place(x=35,y=100,width=80,height=21)
EntryUser = Entry(loginwin)
EntryUser.place(x=140,y=100,width=200,height=25)
Label(loginwin,text=" 密码",font=("宋体",15,"bold")).place(x=35,y=150,width=80,height=21)
EntryPwd = Entry(loginwin,show="-")
EntryPwd.place(x=140,y=150,width=200,height=25)

CheckVar_1 = IntVar()
check_hobby_1 = Checkbutton(loginwin,text="管理员选项",variable=CheckVar_1,onvalue=1,offvalue=0)
check_hobby_1.place(x=160,y=180)

ButtonUser = Button(loginwin,text="登录",command=login).place(x=100,y=250,width=50,height=30)
Buttonquit = Button(loginwin,text="退出",command=loginwin.destroy).place(x=250,y=250,width=50,height=30)
Buttonreg = Button(loginwin,text="注册",command=regist).place(x=175,y=250,width=50,height=30)
lb = Label(loginwin,text="注意，注册时请勿勾选管理员选项",font=("宋体",10,"bold")).place(x=100,y=290)
loginwin.mainloop()