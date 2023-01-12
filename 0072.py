for i in range(1,50):
    for j in range(1,50):
        if((i+j)*(i-j)==68):
            print(i,j)
            break
a = 18*18-168
print("原数字是:>%d"%a)