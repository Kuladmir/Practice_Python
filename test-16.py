import turtle as t
t.setup(800, 600)
t.width(3)
t.color("Blue", (0.14, 0.75, 0.26))
t.begin_fill()
for i in range(3):
    t.forward(100)
    t.left(120)
t.end_fill()
t.forward(100)
t.color("Pink",[0.24, 0.59, 0.74])
t.begin_fill()
for i in range(3):
    t.forward(100)
    t.left(90)
t.end_fill()
t.done()