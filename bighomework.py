from tkinter import *
from tkinter.ttk import *
import tkinter.messagebox
from tkinter.messagebox import *
import pymysql
import tkinter as tk
from tkinter import ttk
def plant():
    win = tk.Toplevel()
    win.title('农产品信息管理平台-农产品信息管理')
    win.geometry('600x400+450+100')
    numLabel = tk.Label(win, text='序列：').grid(row=0, column=0)
    numStringvar = tk.StringVar()
    numEntry = tk.Entry(win, width=20, textvariable=numStringvar).grid(row=0, column=1)
    nameLabel = tk.Label(win, text='作物：').grid(row=1, column=0)
    nameStringvar = tk.StringVar()
    nameEntry = tk.Entry(win, width=20, textvariable=nameStringvar).grid(row=1, column=1)
    nameidLabel = tk.Label(win, text='作物id：').grid(row=2, column=0)
    nameidStringvar = tk.StringVar()
    nameidEntry = tk.Entry(win, width=20, textvariable=nameidStringvar).grid(row=2, column=1)
    whereLabel = tk.Label(win, text='科属：').grid(row=3, column=0)
    whereStringvar = tk.StringVar()
    whereEntry = tk.Entry(win, width=20, textvariable=whereStringvar).grid(row=3, column=1)

    def select():
        xh = numStringvar.get()
        db = pymysql.connect(host='localhost', user='root', password='', db='info', charset='utf8')
        cursor = db.cursor()
        if len(xh) == 0:
            c = cursor.execute('select * from plant')
        else:
            c = cursor.execute('select * from plant where 序列 like "%' + xh + '%"')
        list_re = cursor.fetchall()
        x = tree.get_children()
        for item in x:
            tree.delete(item)
        for i in range(len(list_re)):
            tree.insert('', i, text=i, values=(
                list_re[i][0], list_re[i][1], list_re[i][2], list_re[i][3]))
        db.close()
    selectButton = tk.Button(win, text='查询', width=10, height=1, command=select).grid(row=4, column=0, pady=2)

    def update():
        xh = numStringvar.get()
        db = pymysql.connect(host='localhost', user='root', password='', db='info', charset='utf8')
        cursor = db.cursor()
        sql = "UPDATE plant SET 作物名='" + nameStringvar.get() + "',"+"作物id='"+nameidStringvar.get()+"',"+"科属='"+whereStringvar.get()+"' "+"WHERE 序列 = " + numStringvar.get() + ""
        cursor.execute(sql)
        db.commit()
        tk.messagebox.showinfo('提示', xh + '信息更新成功！')
        db.close()
    UpdateButton = tk.Button(win, text='更新', width=10, height=1, command=update).grid(row=4, column=1, pady=2)

    def insert():
        db = pymysql.connect(host='localhost', user='root', password='', db='info', charset='utf8')
        cursor = db.cursor()
        cursor.execute('insert into plant values ("%s","%s","%s","%s")' % (numStringvar.get(), nameStringvar.get(), nameidStringvar.get(), whereStringvar.get()))
        db.commit()
        tree.insert('', 'end', value=[numStringvar.get(), nameStringvar.get(), nameidStringvar.get(), whereStringvar.get()])
        tk.messagebox.showinfo('提示', numStringvar.get() + '信息插入成功！')
        db.close()
    insertButton = tk.Button(win, text='插入', width=10, height=1, command=insert).grid(row=4, column=2, pady=2)

    def delete():
        xh = numStringvar.get()
        db = pymysql.connect(host='localhost', user='root', password='', db='info', charset='utf8')
        cursor = db.cursor()
        sql = "DELETE FROM plant WHERE 序列='"+xh+"'"
        cursor.execute(sql)
        db.commit()
        tk.messagebox.showinfo('提示', xh + '信息删除成功！')
        db.close()
    deleteButton = tk.Button(win, text='删除', width=10, height=1, command=delete).grid(row=4, column=3, pady=2)

    tree = ttk.Treeview(win)
    tree = Treeview(win)
    tree['column'] = ("序列", "作物名", "作物id", "科属")
    tree.column("序列", width=100)
    tree.column("作物名", width=100)
    tree.column("作物id", width=100)
    tree.column("科属", width=100)
    tree.heading("序列", text="序列")
    tree.heading("作物名", text="作物名")
    tree.heading("作物id", text="作物id")
    tree.heading("科属", text="科属")
    tree.grid(row=5, column=0, columnspan=4)

def data():
    win = tk.Toplevel()
    win.title('农产品信息管理平台-农产品数据管理')
    win.geometry('1000x500+300+100')
    numLabel = tk.Label(win, text='序列：').grid(row=0, column=0)
    numStringvar = tk.StringVar()
    numEntry = tk.Entry(win, width=20, textvariable=numStringvar).grid(row=0, column=1)
    nameLabel = tk.Label(win, text='作物：').grid(row=1, column=0)
    nameStringvar = tk.StringVar()
    nameEntry = tk.Entry(win, width=20, textvariable=nameStringvar).grid(row=1, column=1)
    nameidLabel = tk.Label(win, text='作物id：').grid(row=2, column=0)
    nameidStringvar = tk.StringVar()
    nameidEntry = tk.Entry(win, width=20, textvariable=nameidStringvar).grid(row=2, column=1)
    timeLabel = tk.Label(win, text='检测时间：').grid(row=3, column=0)
    timeStringvar = tk.StringVar()
    timeEntry = tk.Entry(win, width=20, textvariable=timeStringvar).grid(row=3, column=1)
    placeLabel = tk.Label(win, text='检测地点：').grid(row=4, column=0)
    placeStringvar = tk.StringVar()
    placeEntry = tk.Entry(win, width=20, textvariable=placeStringvar).grid(row=4, column=1)
    calLabel = tk.Label(win, text='检测数据：').grid(row=5, column=0)
    calStringvar = tk.StringVar()
    calEntry = tk.Entry(win, width=20, textvariable=calStringvar).grid(row=5, column=1)
    peoLabel = tk.Label(win, text='检测员：').grid(row=6, column=0)
    peoStringvar = tk.StringVar()
    peoEntry = tk.Entry(win, width=20, textvariable=peoStringvar).grid(row=6, column=1)
    scLabel = tk.Label(win, text='评估分数：').grid(row=7, column=0)
    scStringvar = tk.StringVar()
    scEntry = tk.Entry(win, width=20, textvariable=scStringvar).grid(row=7, column=1)

    def select():
        xh = numStringvar.get()
        db = pymysql.connect(host='localhost', user='root', password='', db='info', charset='utf8')
        cursor = db.cursor()
        if len(xh) == 0:
            c = cursor.execute('select * from data')
        else:
            c = cursor.execute('select * from data where 序列 like "%' + xh + '%"')
        list_re = cursor.fetchall()
        x = tree.get_children()
        for item in x:
            tree.delete(item)
        for i in range(len(list_re)):
            tree.insert('', i, text=i, values=(
                list_re[i][0], list_re[i][1], list_re[i][2], list_re[i][3], list_re[i][4], list_re[i][5], list_re[i][6], list_re[i][7]))
        db.close()
    selectButton = tk.Button(win, text='查询', width=10, height=1, command=select).grid(row=8, column=0, pady=2)

    def update():
        xh = numStringvar.get()
        db = pymysql.connect(host='localhost', user='root', password='', db='info', charset='utf8')
        cursor = db.cursor()
        sql = "UPDATE data SET 作物名='"+nameStringvar.get()+"',作物id='"+nameidStringvar.get()+"',检测时间='"+timeStringvar.get()+"',检测地点='"+placeStringvar.get()+"',检测数据='"+calStringvar.get()+"',检测员='"+peoStringvar.get()+"',评估分数='"+scStringvar.get()+"' WHERE 序列 = '" + numStringvar.get() + "'"
        cursor.execute(sql)
        db.commit()
        tk.messagebox.showinfo('提示', xh + '信息更新成功！')
        db.close()
    UpdateButton = tk.Button(win, text='更新', width=10, height=1, command=update).grid(row=8, column=1, pady=2)

    def insert():
        db = pymysql.connect(host='localhost', user='root', password='', db='info', charset='utf8')
        cursor = db.cursor()
        cursor.execute('insert into data values ("%s","%s","%s","%s","%s","%s","%s","%s")' % (numStringvar.get(), nameStringvar.get(), nameidStringvar.get(), timeStringvar.get(), placeStringvar.get(),calStringvar.get(),peoStringvar.get(),scStringvar.get()))
        db.commit()
        tree.insert('', 'end', value=[numStringvar.get(), nameStringvar.get(), nameidStringvar.get(), timeStringvar.get(), placeStringvar.get(),calStringvar.get(),peoStringvar.get(),scStringvar.get()])
        tk.messagebox.showinfo('提示', numStringvar.get() + '信息插入成功！')
        db.close()
    insertButton = tk.Button(win, text='插入', width=10, height=1, command=insert).grid(row=8, column=2, pady=2)

    def delete():
        xh = numStringvar.get()
        db = pymysql.connect(host='localhost', user='root', password='', db='info', charset='utf8')
        cursor = db.cursor()
        sql = "DELETE FROM data WHERE 序列='"+xh+"'"
        cursor.execute(sql)
        db.commit()
        tk.messagebox.showinfo('提示', xh + '信息删除成功！')
        db.close()
    deleteButton = tk.Button(win, text='删除', width=10, height=1, command=delete).grid(row=8, column=3, pady=2)

    tree = ttk.Treeview(win)
    tree = Treeview(win)
    tree['column'] = ("序列", "作物名", "作物id", "检测时间","检测地点","检测数据","检测员","评估分数")
    tree.column("序列", width=100)
    tree.column("作物名", width=100)
    tree.column("作物id", width=100)
    tree.column("检测时间", width=100)
    tree.column("检测地点", width=100)
    tree.column("检测数据", width=100)
    tree.column("检测员", width=100)
    tree.column("评估分数", width=100)
    tree.heading("序列", text="序列")
    tree.heading("作物名", text="作物名")
    tree.heading("作物id", text="作物id")
    tree.heading("检测时间", text="检测时间")
    tree.heading("检测地点", text="检测地点")
    tree.heading("检测数据", text="检测数据")
    tree.heading("检测员", text="检测员")
    tree.heading("评估分数", text="评估分数")
    tree.grid(row=9, column=0, columnspan=4)

def referdata():
    win = tk.Toplevel()
    win.title('农产品信息管理平台-数据更新')
    win.geometry('700x400+350+100')
    numLabel = tk.Label(win, text='序列：').grid(row=0, column=0)
    numStringvar = tk.StringVar()
    numEntry = tk.Entry(win, width=20, textvariable=numStringvar).grid(row=0, column=1)
    nameLabel = tk.Label(win, text='作物：').grid(row=1, column=0)
    nameStringvar = tk.StringVar()
    nameEntry = tk.Entry(win, width=20, textvariable=nameStringvar).grid(row=1, column=1)
    nameidLabel = tk.Label(win, text='检测对象：').grid(row=2, column=0)
    nameidStringvar = tk.StringVar()
    nameidEntry = tk.Entry(win, width=20, textvariable=nameidStringvar).grid(row=2, column=1)
    dataLabel = tk.Label(win, text='参考数据：').grid(row=3, column=0)
    dataStringvar = tk.StringVar()
    dataEntry = tk.Entry(win, width=20, textvariable=dataStringvar).grid(row=3, column=1)
    scLabel = tk.Label(win, text='参考评分：').grid(row=4, column=0)
    scStringvar = tk.StringVar()
    scEntry = tk.Entry(win, width=20, textvariable=scStringvar).grid(row=4, column=1)

    def select():
        xh = numStringvar.get()
        db = pymysql.connect(host='localhost', user='root', password='', db='info', charset='utf8')
        cursor = db.cursor()
        if len(xh) == 0:
            c = cursor.execute('select * from referdata')
        else:
            c = cursor.execute('select * from referdata where 序列 like "%' + xh + '%"')
        list_re = cursor.fetchall()
        x = tree.get_children()
        for item in x:
            tree.delete(item)
        for i in range(len(list_re)):
            tree.insert('', i, text=i, values=(
                list_re[i][0], list_re[i][1], list_re[i][2], list_re[i][3],list_re[i][4]))
        db.close()
    selectButton = tk.Button(win, text='查询', width=10, height=1, command=select).grid(row=5, column=0, pady=2)

    def update():
        xh = numStringvar.get()
        db = pymysql.connect(host='localhost', user='root', password='', db='info', charset='utf8')
        cursor = db.cursor()
        sql = "UPDATE referdata SET 作物名='"+nameStringvar.get()+"',"+"检测对象='"+nameidStringvar.get()+"',"+"参考数据='"+dataStringvar.get()+"',"+"参考评分='"+scStringvar.get()+"' "+"WHERE 序列 = '" + numStringvar.get() + "'"
        cursor.execute(sql)
        db.commit()
        tk.messagebox.showinfo('提示', xh + '信息更新成功！')
        db.close()
    UpdateButton = tk.Button(win, text='更新', width=10, height=1, command=update).grid(row=5, column=1, pady=2)

    def insert():
        db = pymysql.connect(host='localhost', user='root', password='', db='info', charset='utf8')
        cursor = db.cursor()
        cursor.execute('insert into referdata values ("%s","%s","%s","%s",%s)' % (numStringvar.get(), nameStringvar.get(), nameidStringvar.get(), dataStringvar.get(), scStringvar.get()))
        db.commit()
        tree.insert('', 'end', value=[numStringvar.get(), nameStringvar.get(), nameidStringvar.get(), dataStringvar.get(), scStringvar.get()])
        tk.messagebox.showinfo('提示', numStringvar.get() + '信息插入成功！')
        db.close()
    insertButton = tk.Button(win, text='插入', width=10, height=1, command=insert).grid(row=5, column=2, pady=2)

    def delete():
        xh = numStringvar.get()
        db = pymysql.connect(host='localhost', user='root', password='', db='info', charset='utf8')
        cursor = db.cursor()
        sql = "DELETE FROM referdata WHERE 序列='"+xh+"'"
        cursor.execute(sql)
        db.commit()
        tk.messagebox.showinfo('提示', xh + '信息删除成功！')
        db.close()
    deleteButton = tk.Button(win, text='删除', width=10, height=1, command=delete).grid(row=5, column=3, pady=2)

    tree = ttk.Treeview(win)
    tree = Treeview(win)
    tree['column'] = ("序列", "作物名", "检测对象", "参考数据", "参考评分")
    tree.column("序列", width=100)
    tree.column("作物名", width=100)
    tree.column("检测对象", width=100)
    tree.column("参考数据", width=100)
    tree.column("参考评分", width=100)
    tree.heading("序列", text="序列")
    tree.heading("作物名", text="作物名")
    tree.heading("检测对象", text="检测对象")
    tree.heading("参考数据", text="参考数据")
    tree.heading("参考评分", text="参考评分")
    tree.grid(row=6, column=0, columnspan=4)

def login():
    root_window = tk.Tk()
    root_window.title('农产品信息管理平台')
    root_window.geometry('500x500+500+100')
    label = tk.Label(root_window, text="欢迎访问\n\n农产品信息管理新平台", font=('宋体', 20), bg="#21ECE5",width=20, height=5,padx=5, pady=10, borderwidth=10, relief="sunken")
    label.pack()
    root_window.config(background="#3EF298")
    button = tk.Button(root_window, text="点击这里前往农产品信息管理", command=plant,width = 30)
    button.pack(padx = 50,pady = 22)
    button = tk.Button(root_window, text="点击这里前往农产品数据管理", command=data,width = 30)
    button.pack(padx = 50,pady = 22)
    button = tk.Button(root_window, text="点击这里前往农产品标准数据管理", command=referdata,width = 30)
    button.pack(padx = 50,pady = 22)
    button = tk.Button(root_window, text="点击这里关闭页面", command=root_window.quit,width = 20)
    button.pack(padx = 50,pady = 22)
    root_window.mainloop()

def original_window():
    enter_w = tk.Tk()
    enter_w.title("农产品信息管理平台")
    enter_w.geometry('400x200+500+300')
    lab_1 = tk.Label(enter_w, width=7, text='账号')
    lab_1.place(x=80, y=50)
    lab_2 = tk.Label(enter_w, width=7, text='密码')
    lab_2.place(x=80, y=80)
    global uesr_name, password
    user_name = tk.StringVar()
    password = tk.StringVar()
    entry = tk.Entry(enter_w, textvariable=user_name)
    entry.pack()
    entry.place(x=150, y=50)
    entry_1 = tk.Entry(enter_w, show="*", textvariable=password)
    entry_1.pack()
    entry_1.place(x=150, y=80)
    def panduan(enter_w):
        db = pymysql.connect(host='localhost', user='root', password='', db='info', charset='utf8')
        cursor = db.cursor()
        cursor.execute('SELECT 密码 FROM userlist WHERE 账号 = %s'%entry.get())
        data = cursor.fetchall()
        if data:
            if entry_1.get() == (data[0])[0]:
                tk.messagebox.showinfo('^_^', '密码正确,欢迎使用本软件')
                login()
        else:
            tk.messagebox.showerror('*_*', '账号或密码错误,请重新输入')
        db.close()
    btn = tk.Button(enter_w, text='登陆', fg="black", width=7, compound='center', \
                    bg="white", command=lambda: panduan(enter_w))
    btn.pack()
    btn.place(x=170, y=120)
    enter_w.mainloop()

original_window()
