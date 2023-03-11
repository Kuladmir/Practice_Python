from tkinter import Tk
from tkinter import messagebox

root = Tk()
def qws():
    if messagebox.showwaring("Error","出现错误"):
        root.destroy()
        
root.protocol("WM_DELETE_WINDOW",qws)
root.mainloop()