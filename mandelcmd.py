from __future__ import print_function

import sys
import tty
from decimal import Decimal, getcontext
from json import dumps, loads
from math import floor, log10, sqrt
from os import mkdir, path, popen, remove, system
from time import sleep

from imageio import get_writer, imread, imwrite, mimsave
from matplotlib import cm
from numba import autojit, jit, prange
from numpy import (array, clip, empty, fliplr, float64, linspace, logspace,
                   percentile, ravel, remainder, swapaxes, uint8, uint16,
                   zeros)
from pick import pick
from PIL import Image, ImageFilter
from tqdm import tqdm

from dependencies.colors import cols, noiseColor
from dependencies.functions import (bookmarks, juliafast, juliaimg, juliapan,
                                    mandelanimrender, mandelfast, mandelimg,
                                    ndec)


@jit
def printiters(iters, colors):
    '''literally just a seperate function to print colors to the command line'''
    for i in iters:
        print(u'\u001b[48;2;' + str(colors[i, 0]) + ';' + str(colors[i, 1]) + ';' + str(colors[i, 2]) + 'm ', end='')

def mandelloop():
    '''main loop for the command line drawing'''
    system('printf \\\\033c')  # clearing the screen, works faster than calling system("clear")
    tty.setraw(sys.stdin)  # preparin to take the raw keyboard input
    # initiating the starting vars
    cx = -.5
    cy = 0.0
    w = 4.5
    maxiters = 100
    waypoints = []
    choice = 0
    noise = noiseColor()

    while 1:  # main loop
        rows, columns = popen('stty size', 'r').read().split()  # getting size
        height = int(rows) - 1  # setting vars for size
        width = int(columns)

        # correcting aspect ratio, command line output is almost exactly a 1:2 aspect ratio
        h = (w * height * 2) / width
        # setting min and max values for calling mandelfast
        minx = cx - .5 * w
        maxx = cx + .5 * w
        miny = cy - .5 * h
        maxy = cy + .5 * h

        # getting per character iterations for printing to console
        iters = swapaxes(mandelfast(minx, maxx, miny, maxy, maxiters, width, height), 0, 1).ravel() # turns iters into a 1-d array for les for loop nesting
        if choice == 0: # off sets iters to the bottom value for grayscale output
            iters -= iters.min()
        iters += 1
        iters %= iters.max()

        colors = cols(choice, iters.max() + 1, noise) # gets color list for fast printing
        sys.stdout.write(u'\u001b[0;1f')  # resetting cursor to top left
        printiters(iters, colors) # calling compiled function for printing so the only holdup is print speed
        # making coordinate output to display location to user
        coords = u"\u001b[0mX: " + str(ndec(cx, 5)) + " " * 5 + "Y: " + str(ndec(cy, 5)) + " " * \
            5 + "Width: " + str(ndec(w, 5)) + " " * 5 + \
            "Iters: " + str(maxiters) + u"\u001b[0K"
        print(coords, end=' ' * 5) # prints coords, width and iterations
        if len(waypoints) == 1: # lets user know if they already have a start point for julia pan animation
            print("pick second point", end='')
        key = sys.stdin.read(1)                                                         # gets input character for parsing
        if ord(key) == 3: break                                     			        # taking raw input so breaking if escape sequence is used
        elif key == 'a': cx -= w * .1 											        # getting direction inputs
        elif key == 'A': cx -= w * .05 											        # getting direction inputs
        elif key == 'd': cx += w * .1 											        # getting direction inputs
        elif key == 'D': cx += w * .05 											        # getting direction inputs
        elif key == 'w': cy -= w * .1 											        # getting direction inputs
        elif key == 'W': cy -= w * .05 											        # getting direction inputs
        elif key == 's': cy += w * .1 											        # getting direction inputs
        elif key == 'S': cy += w * .05 											        # getting direction inputs
        elif key == 'q': w *= .9      											        # getting zoom inputs
        elif key == 'Q': w *= .95      											        # getting zoom inputs
        elif key == 'e': w *= 1.1     											        # getting zoom inputs
        elif key == 'E': w *= 1.05     											        # getting zoom inputs
        elif key == 'z': maxiters *= .9; maxiters = int(maxiters)   			        # decreasing iters
        elif key == 'Z': maxiters *= .95; maxiters = int(maxiters)   			        # decreasing iters
        elif key == 'c': maxiters *= 1.1; maxiters = int(maxiters)  			        # increasing iters
        elif key == 'C': maxiters *= 1.05; maxiters = int(maxiters)  			        # increasing iters
        elif key.lower() == 'v': mandelanimrender(cx, cy, w, maxiters, choice, noise)   # calling anim output for current point
        elif key.lower() == 'i':                                                        # getting pushing current coords to array for julia pan
            waypoints.append((cx, cy))
            if len(waypoints) > 1: # if this is the second point calls julia pan with coord list
                juliapan(waypoints, choice, noise)
                waypoints = []     # resets coord list
        elif key.lower() == 'm': choice += 1; choice %= 3                               # switches between the 3 color modes
        elif key.lower() == 'n': noise.newcolors()                                      # randomizes colors for noise
        elif key.lower() == ' ': julialoop(cx, cy, choice, noise)     			        # calling anim output for current point
        elif key.lower() == 'r': cx = -.5; cy = 0; w = 5.0; maxiters = 100  	        # resetting to default values
        elif key.lower() == 'x': 		
            try:										                                # getting user input for coords to jump to
                system('reset') # resets console to print input
                cx = float(raw_input('CenterX: '))
                cy = float(raw_input('CenterY: '))
            except ValueError:
                pass
            tty.setraw(sys.stdin) # sets console back to raw to recieve input
        elif key.lower() == 'b':                                                        # opens bookmark window
            cx, cy, w, maxiters = bookmarks(cx, cy, w, maxiters)
            tty.setraw(sys.stdin)
        elif key.lower() == 'f':                                                        # calling the menu to choose resolution for image output
            mandelimg(cx, cy, w, maxiters, choice, noise)
        else:
            pass														                # if not an accepted input continues loop
        sys.stdout.flush()
    system('reset') # sets console back to original state

def julialoop(mandelx, mandely, choice, noise):
    '''main loop for the command line drawing of the julia set'''
    system('printf \\\\033c')  # clearing the screen, works faster than calling system("clear")
    tty.setraw(sys.stdin)  # preparin to take the raw keyboard input
    # initiating the starting vars
    cx = 0.0
    cy = 0.0
    w = 4.0
    maxiters = 100

    while 1:  # main loop
        rows, columns = popen('stty size', 'r').read().split()  # getting size
        height = int(rows)  # setting vars for size
        width = int(columns) + 1
        # correcting aspect ratio, command line output is almost exactly a 1:2 aspect ratio
        h = (w * height * 2) / width
        # setting min and max values for calling mandelfast
        minx = cx - .5 * w
        maxx = cx + .5 * w
        miny = cy - .5 * h
        maxy = cy + .5 * h

        # getting per character iterations for printing to console
        iters = juliafast(minx, maxx, miny, maxy, mandelx, mandely, maxiters, width, height)
        if choice == 0:
            iters -= iters.min()
        iters %= iters.max()
        maximum = iters.max() + 1

        colors = cols(choice, maximum + 1, noise)

        sys.stdout.write(u'\u001b[0;1f')  # resetting cursor to top left
        for y in range(1, height):
            for x in range(1, width):
                print(u'\u001b[48;2;' + str(colors[iters[x, y], 0]) + ';' + str(colors[iters[x, y], 1]) + ';' + str(colors[iters[x, y], 2]) + 'm ', end='')
        coords = u"\u001b[0mX: " + str(ndec(cx, 5)) + " " * 5 + "Y: " + str(ndec(cy, 5)) + " " * \
            5 + "Width: " + str(ndec(w, 5)) + " " * 5 + \
            "Iters: " + str(maxiters) + u"\u001b[0K"
        print(coords, end='') # same as mandelbrot loop, prints coords, width and iterations to the bottom left of the screen

        key = sys.stdin.read(1)
        if ord(key) == 3: break														# taking raw input so breaking if escape sequence is used
        elif key == 'a': cx -= w * .1 												# getting direction inputs
        elif key == 'A': cx -= w * .05 												# getting direction inputs
        elif key == 'd': cx += w * .1 												# getting direction inputs
        elif key == 'D': cx += w * .05 												# getting direction inputs
        elif key == 'w': cy -= w * .1 												# getting direction inputs
        elif key == 'W': cy -= w * .05 												# getting direction inputs
        elif key == 's': cy += w * .1 												# getting direction inputs
        elif key == 'S': cy += w * .05 												# getting direction inputs
        elif key == 'q': w *= .9      												# getting zoom inputs
        elif key == 'Q': w *= .95      												# getting zoom inputs
        elif key == 'e': w *= 1.1     												# getting zoom inputs
        elif key == 'E': w *= 1.05     												# getting zoom inputs
        elif key == 'z': maxiters *= .9; maxiters = int(maxiters)   				# decreasing iters
        elif key == 'Z': maxiters *= .95; maxiters = int(maxiters)   				# decreasing iters
        elif key == 'c': maxiters *= 1.1; maxiters = int(maxiters)  				# increasing iters
        elif key == 'C': maxiters *= 1.05; maxiters = int(maxiters)  				# increasing iters
        elif key.lower() == 'n': noise.newcolors()                                  # regenerates colors for noise coloration
        elif key.lower() == 'r': cx = 0; cy = 0; w = 4.0; maxiters = 100  			# resetting to default values
        elif key.lower() == 'f': juliaimg(cx, cy, mandelx, mandely, w, maxiters, choice, noise) # calls function to render current julia set as image
        elif key.lower() == ' ': break                                              # returns to the mandelbrot set loop
        else:
            pass																	# if not an accepted input continues loop
        sys.stdout.flush()

if __name__ == '__main__':
    if not path.isdir("./images/"): mkdir("./images/") # checks if needed folders exist for image and animation output
    if not path.isdir("./zoomcache/"): mkdir("./zoomcache/")
    if not path.isdir("./itercache/"): mkdir("./itercache/")
    if not path.isdir("./janimcache/"): mkdir("./janimcache/")
    if not path.isdir("./anim/"): mkdir("./anim/")
    mandelloop() # starts main loop
