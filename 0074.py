import tkinter as tk
root_windows = tk.Tk()
root_windows.title("开放的CAU")
root_windows.geometry("450x300")
root_windows.resizable(0,0)
root_windows.iconbitmap('./image/KuGouMusic.ico')
root_windows["background"] = "#c9c9c9"
text = tk.Label(root_windows,text="Kuladmir",bg="yellow",fg="red",font=("Times",24,"bold italic"))
text.pack()
button = tk.Button(root_windows,text="退出",command=root_windows.quit)
button.pack(side="bottom")
root_windows.mainloop()

