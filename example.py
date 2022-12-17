import math

B = 100
w = 1
dt = 0.01
N = 1001


def f_x(t):
    return B * math.cos(w * t)


with open("fs.dat", 'w') as fs:
    for i in range(1001):
        for j in range(12):
            if j == 8:
                fs.write(f'{f_x(i * dt)} ')
            else:
                fs.write('0 ')
        fs.write('\n')
