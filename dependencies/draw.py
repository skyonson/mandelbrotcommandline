from __future__ import print_function
setpos = lambda x, y: print(u"\u001b[" + str(y) + ';' + str(x) + 'H', end='')

point = lambda x, y: print(u"\u001b[" + str(y) + ';' + str(x) + 'H ', end='')

background = lambda r, g, b: print(u"\u001b[48;2;" + str(r) + ';'+ str(g) + ';'+ str(b) + 'm', end='')

foreground = lambda r, g, b: print(u"\u001b[38;2;" + str(r) + ';'+ str(g) + ';'+ str(b) + 'm', end='')

def box(x1, y1, x2, y2):
    for i in range(x1, x2):
        point(i, y1)
        point(i, y2)
    for i in range(y1, y2):
        point(x1, i)
        point(x2, i)