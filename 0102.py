import tkinter as tk
window = tk.Tk()
window.title("购物车界面")
window.geometry('450x200+450+250')
window.resizable(0,0)
# 创建一个文本标签
label = tk.Label(window, bg='#9FB6CD',width=18, text='')
label.grid(row =2)
# 创建执行函数
def select_price(value):
 label.config(text='您购买的数量是 ' + value)
# 创建 Scale 控件
scale = tk.Scale(window,
 label='选择您要购买的数量',
 from_=1,
 to= 100,
 orient=tk.HORIZONTAL, # 设置 Scale 控件平方向显示
 length=400,
 tickinterval=9, # 设置刻度滑动条的间隔
 command=select_price) # 调用执行函数，是数值显示在 Label 控件中
scale.grid(row =1)
# 显示窗口
window.mainloop()
