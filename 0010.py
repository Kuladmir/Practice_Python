a = int(input("请输入你的年龄"))
if a>=18:
    print("恭喜，你已成年")
    print("你可以解除防沉迷系统")
    print("但请注意休息")
elif a < 18 and a >= 16:
    print("你未成年，玩游戏受到显示")
else:
    print("你不被允许加入游戏")
