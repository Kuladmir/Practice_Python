def result1():
 # re.delete("0.0",END)
 print(v.get())
 if v.get() == 0: #选择“一定会”的答案解析
  str="结果：\n 很遗憾，你未能选对正确的答案。送你一句诗词：人生得意须尽欢，莫使金樽空对月。"
 elif v.get() == 1: #选择“很可能会”的答案解析
  str="结果：\n 恭喜，你选对了。送你一句诗词：一帆离人画，一屏愁风卷"
 elif v.get() == 2: #选择“偶尔会”的答案解析
  str="结果：\n 很遗憾，你未能选对正确的答案。送你一句诗词：山有木兮木有枝，心悦君兮君不知。"
 else: #选择“绝不会”的答案解析
  str="结果：\n 很遗憾，你未能选对正确的答案。送你一句诗词：羽毛不足辱弹射，滋味不足登俎豆。 "
 re.config(text=str)

from tkinter import *
win = Tk()
win.title("一个简单的心理测试题")
win.geometry("400x500")
# 数组存储单选按钮显示的值
str1 = ["正方形", "矩形", "三角形", "圆形"]
Label(win, text="你能否找对那个正确的图形，并得到一句原创诗词", font="14").pack(anchor=W)
text = Label(win, text="注意：这个答案可能不是圆形",
font="14").pack(anchor=W)
v = IntVar() #该变量绑定一组单选按钮的值
for i in range(len(str1)):
 # text 为单选按钮旁显示的文字,value 为单选按钮的值，indicatoron 设置单选按钮为矩形，selectcolor 设置被选中的颜色
 radio = Radiobutton(win, text=str1[i], variable=v, value=i, font="12", indicatoron=0,
selectcolor="#00ffff")
 radio.pack(side=TOP, fill=X, padx=20, pady=3)
#提交按钮
button = Button(win, text="提交", command=result1, font="14", bg="#4CC6E3")
button.pack(side=TOP, fill=X, padx=40, pady=20)
# 显示答案解析的 Label 组件
re = Label(win, font="14", height="10", width="40", justify="left",wraplength=320)
re.pack(side=TOP, pady=10)
win.mainloop()
