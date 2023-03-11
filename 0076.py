import tkinter
from tkinter import ttk
root = tkinter.Tk()
root.title("Hello World")
ttk.Style().configure("TButton",font=20,relief="flat",background="#00ffff")
btn = ttk.Button(text="按钮测试",style="TButton").pack(pady=20)
root.mainloop()
