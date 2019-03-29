from random import random

from matplotlib import cm
from noise import pnoise1
from numba import jit
from numpy import uint8, zeros

from java import remap


@jit(parallel=True, nogil=True)
def colmap(c): return int(remap(c, 0, 1, 0, 255)) # just used to swap gradient from decimals to regular color values


def cols(choice, its, noise, miniters = 0, highlight = -1):
    '''Returns an array of colors mapped to iteration by index'''
    colors = zeros((its, 3), uint8)
    # switches between highlighting individual iterations or just giving a color map
    if highlight == -1:
        colors[0] = (0, 0, 0)
        if choice == 0:
            for i in range(miniters, its): colors[i] = (int(remap(i, miniters, its, 0.0, 255.0)), int(remap(i, miniters, its, 0.0, 255.0)), int(remap(i, miniters, its, 0.0, 255.0)))
            return colors
        if choice == 1:
            for i in range(1, its): colors[i] = (colmap(cm.gnuplot2(abs(remap(((i + 256) % 512), 0, 512, -1.0, 1.0)))[0]), colmap(cm.gnuplot2(abs(remap(((i + 256) % 512), 0, 512, -1.0, 1.0)))[1]), colmap(cm.gnuplot2(abs(remap(((i + 256) % 512), 0, 512, -1.0, 1.0)))[2]))
            return colors
        if choice == 2:
            for i in range(1, its): colors[i] = noise.gencolor(i)
            return colors
    else:
        if choice == 0:
            colors[highlight] = (255, 255, 255)
            return colors
        if choice == 1:
            colors[highlight] = (colmap(cm.gnuplot2(abs(remap(((highlight + 256) % 512), 0, 512, -1.0, 1.0)))[0]), colmap(cm.gnuplot2(abs(remap(((highlight + 256) % 512), 0, 512, -1.0, 1.0)))[1]), colmap(cm.gnuplot2(abs(remap(((highlight + 256) % 512), 0, 512, -1.0, 1.0)))[2]))
            return colors
        if choice == 2:
            colors[highlight] = noise.gencolor(highlight)
            return colors

class noiseColor:
    def __init__(self):
        # initializes RGB to random offsets
        self.rand1 = random()
        self.rand2 = random()
        self.rand3 = random()
        self.scale = 100.0

    def newcolors(self):
        # Reinitializes RGB offsets
        self.rand1 = random()
        self.rand2 = random()
        self.rand3 = random()

    def gencolor(self, n):
        '''Returns (R, G, B) for the chosen iteration'''
        n /= self.scale
        return (
        int(abs(pnoise1(((n + self.rand1) % 1) ** 2, octaves=3, repeat=8192)) * 255),
        int(abs(pnoise1(((n + self.rand2) % 1) ** 2, octaves=3, repeat=8192)) * 255),
        int(abs(pnoise1(((n + self.rand3) % 1) ** 2, octaves=3, repeat=8192)) * 255)
        )
