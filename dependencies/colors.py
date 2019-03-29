from noise import pnoise1
from random import random
class noiseColor:
    def __init__(self):
        self.rand1 = random()
        self.rand2 = random()
        self.rand3 = random()
        self.scale = 100.0

    def newcolors(self):
        self.rand1 = random()
        self.rand2 = random()
        self.rand3 = random()

    def gencolor(self, n):
        n /= self.scale
        return (
        int(abs(pnoise1(((n + self.rand1) % 1) ** 2, octaves=3, repeat=8192)) * 255),
        int(abs(pnoise1(((n + self.rand2) % 1) ** 2, octaves=3, repeat=8192)) * 255),
        int(abs(pnoise1(((n + self.rand3) % 1) ** 2, octaves=3, repeat=8192)) * 255)
        )
