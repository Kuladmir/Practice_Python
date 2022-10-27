for b in range(100, 1000, 1):
    a = b
    d = 0
    while int(a) != 0:
        c = int(a) % 10
        a = int(a)/10
        d = d + c*c*c
        if d == b and int(a) < 10:
            print(d)

