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

from dependencies.colors import noiseColor
from dependencies.java import remap


def cols(choice, its, noise, miniters = 0, highlight = -1):
    colors = zeros((its, 3), uint16)
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

@jit(parallel=True, nopython=True, nogil=True)
def mandelfact(x, y, iterations):
    '''does mandelbrot factorization for a given point up to a certain number of iterations'''
    q = ((x - .25) ** 2) + (y ** 2)
    if q * (q + x - .25) < .25 * y ** 2:
        return iterations
    if (x + 1) ** 2 + y ** 2 <= 1/16.0:
        return iterations
    ca = x
    cb = y
    n = 1
    while n < iterations and x * x + y * y < 4.0:
        aa = x * x - y * y
        bb = 2 * x * y
        x = aa + ca
        y = bb + cb
        n += 1
    return n

@jit(parallel=True, nopython=True, nogil=True)
def juliafact(x, y, mx, my, iterations):
    '''does julia factorization for a given point up to a certain number of iterations'''
    n = 1
    while n < iterations and x * x + y * y < 4.0:
        aa = x * x - y * y
        bb = 2 * x * y
        x = aa + mx
        y = bb + my
        n += 1
    return n

@jit(parallel=True, nogil=True)
def mandelfast(minx, maxx, miny, maxy, maxiter, width, height):
    '''returns an \'antialiased\' list of iters'''
    # initializing arrays for
    iterxl = empty((width * 2, height * 2), uint16)  # larger iteration array
    iters = empty((width, height), uint16)			# final iteration array
    xspace = linspace(minx, maxx, width * 2, dtype=float64)		# faster to access xcoords
    yspace = linspace(miny, maxy, height * 2, dtype=float64)		# faster to access ycoords

    for x in range(width * 2):
        for y in range(height * 2):
            # generating initial iterations
            iterxl[x, y] = mandelfact(xspace[x], yspace[y], maxiter)

    for x in range(width):
        for y in range(height):
            x2 = x * 2
            y2 = y * 2
            iters[x, y] = int(iterxl[x2, y2] +
                              iterxl[x2 + 1, y2] +
                              iterxl[x2 + 1, y2 + 1] +
                              iterxl[x2, y2 + 1] / 4)  # quick and dirty antialiasing just averages nearby values
    return iters

@jit(parallel=True, nogil=True)
def juliafast(minx, maxx, miny, maxy, mandelx, mandely, maxiter, width, height):
    '''returns an \'antialiased\' list of iters'''
    # initializing arrays for
    iterxl = empty((width * 2, height * 2), uint16)  # larger iteration array
    iters = empty((width, height), uint16)			# final iteration array
    xspace = linspace(minx, maxx, width * 2, dtype=float64)		# faster to access xcoords
    yspace = linspace(miny, maxy, height * 2, dtype=float64)		# faster to access ycoords

    for x in range(width * 2):
        for y in range(height * 2):
            # generating initial iterations
            iterxl[x, y] = juliafact(xspace[x], yspace[y], mandelx, mandely, maxiter)

    for x in range(width):
        for y in range(height):
            x2 = x * 2
            y2 = y * 2
            iters[x, y] = int(iterxl[x2, y2] +
                              iterxl[x2 + 1, y2] +
                              iterxl[x2 + 1, y2 + 1] +
                              iterxl[x2, y2 + 1] / 4)  # quick and dirty antialiasing just averages nearby values
    return iters

@jit
def printiters(iters, colors):
    for i in iters:
        print(u'\u001b[48;2;' + str(colors[i, 0]) + ';' + str(colors[i, 1]) + ';' + str(colors[i, 2]) + 'm ', end='')

def mandelloop():
    '''main loop for the command line drawing'''
    system('printf \\\\033c')  # clearing the screen, works faster than calling system("clear")
    tty.setraw(sys.stdin)  # preparin to take the raw keyboard input
    # initiating the starting vars
    cx = -.5
    cy = 0.0
    w = 5.0
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
        iters = swapaxes(mandelfast(minx, maxx, miny, maxy, maxiters, width, height), 0, 1).ravel()
        if choice == 0:
            iters -= iters.min()
        iters += 1
        iters %= iters.max()

        colors = cols(choice, iters.max() + 1, noise)
        sys.stdout.write(u'\u001b[0;1f')  # resetting cursor to top left
        printiters(iters, colors)

        # for y in range(1, height):
        # 	for x in range(1, width):
        # 		print(u'\u001b[48;2;' + str(colors[iters[x, y], 0]) + ';' + str(colors[iters[x, y], 1]) + ';' + str(colors[iters[x, y], 2]) + 'm ', end='')
        # making coordinate output to display location to user
        coords = u"\u001b[0mX: " + str(ndec(cx, 5)) + " " * 5 + "Y: " + str(ndec(cy, 5)) + " " * \
            5 + "Width: " + str(ndec(w, 5)) + " " * 5 + \
            "Iters: " + str(maxiters) + u"\u001b[0K"
        print(coords, end=' ' * 5)
        if len(waypoints) == 1:
            print("pick second point", end='')
        key = sys.stdin.read(1)
        if ord(key) == 3: break                                     			# taking raw input so breaking if escape sequence is used
        elif key == 'a': cx -= w * .1 											# getting direction inputs
        elif key == 'A': cx -= w * .05 											# getting direction inputs
        elif key == 'd': cx += w * .1 											# getting direction inputs
        elif key == 'D': cx += w * .05 											# getting direction inputs
        elif key == 'w': cy -= w * .1 											# getting direction inputs
        elif key == 'W': cy -= w * .05 											# getting direction inputs
        elif key == 's': cy += w * .1 											# getting direction inputs
        elif key == 'S': cy += w * .05 											# getting direction inputs
        elif key == 'q': w *= .9      											# getting zoom inputs
        elif key == 'Q': w *= .95      											# getting zoom inputs
        elif key == 'e': w *= 1.1     											# getting zoom inputs
        elif key == 'E': w *= 1.05     											# getting zoom inputs
        elif key == 'z': maxiters *= .9; maxiters = int(maxiters)   			# decreasing iters
        elif key == 'Z': maxiters *= .95; maxiters = int(maxiters)   			# decreasing iters
        elif key == 'c': maxiters *= 1.1; maxiters = int(maxiters)  			# increasing iters
        elif key == 'C': maxiters *= 1.05; maxiters = int(maxiters)  			# increasing iters
        elif key.lower() == 'v': mandelanimrender(cx, cy, w, maxiters, choice, noise)   # calling anim output for current point
        elif key.lower() == 'i':
            waypoints.append((cx, cy))
            if len(waypoints) > 1:
                juliapan(waypoints, choice, noise)
                waypoints = []
        elif key.lower() == 'm': choice += 1; choice %= 3
        elif key.lower() == 'n': noise.newcolors()
        elif key.lower() == ' ': julialoop(cx, cy, choice, noise)     			# calling anim output for current point
        elif key.lower() == 'r': cx = -.5; cy = 0; w = 5.0; maxiters = 100  	# resetting to default values
        elif key.lower() == 'x': 		
            try:										# getting user input for coords, might need some tweaking to allow fixing mistakes
                system('reset')
                cx = float(raw_input('CenterX: '))
                cy = float(raw_input('CenterY: '))
            except ValueError:
                pass
            tty.setraw(sys.stdin)
        elif key.lower() == 'b':
            cx, cy, w, maxiters = bookmarks(cx, cy, w, maxiters)
            tty.setraw(sys.stdin)
        elif key.lower() == 'f':
            # calling the menu to choose resolution for image output
            mandelimg(cx, cy, w, maxiters, choice, noise)
        else:
            pass														# if not an accepted input continues loop
        sys.stdout.flush()
    # used to fix the console after setting to raw input
    system('reset')

def mandelimg(cx, cy, w, iters, colchoice, noise):
    # used to call mandelimggen with all variables, also provides selection to choose resolution
    print(u'\u001b[0m' + u'\u001b[1000D' + u'\u001b[1000A')
    title = "Pick a resolution for the rendered image:"
    options = [(1920, 1080), (2560, 1440), (3840, 2160),
               (7680, 4320), 'Custom', 'Cancel']
    choice = pick(options, title)[0]
    if choice == 'Custom':
        try:
            system('reset')
            width = int(raw_input("Width: "))
            height = int(raw_input("Height: "))
            aa = int(raw_input("AA Samples: "))
            mandelimggen(cx, cy, w, iters, (width, height), colchoice, noise, aa)
        except ValueError:
            print("Input numbers only", end = '')
            sleep(1)
    elif choice == 'Cancel':
        pass
    else:
        try:
            system('reset')
            aa = int(raw_input("AA Samples: "))
            mandelimggen(cx, cy, w, iters, choice, colchoice, noise, aa)
        except ValueError:
            print("Input numbers only", end = '')
            sleep(1)
    tty.setraw(sys.stdin)

def juliaimg(cx, cy, mx, my, w, iters, colchoice, noise):
    # used to call mandelimggen with all variables, also provides selection to choose resolution
    print(u'\u001b[0m' + u'\u001b[1000D' + u'\u001b[1000A')
    title = "Pick a resolution for the rendered image:"
    options = [(1920, 1080), (2560, 1440), (3840, 2160),
               (7680, 4320), 'Custom', 'Cancel']
    choice = pick(options, title)[0]
    if choice == 'Custom':
        try:
            system('reset')
            width = int(raw_input("Width: "))
            height = int(raw_input("Height: "))
            aa = int(raw_input("AA Samples: "))
            juliaimggen(cx, cy, mx, my, w, iters, (width, height), colchoice, noise, aa)
        except ValueError:
            print("Input numbers only", end = '')
            sleep(1)
    elif choice == 'Cancel':
        pass
    else:
        try:
            system('reset')
            aa = int(raw_input("AA Samples: "))
            juliaimggen(cx, cy, mx, my, w, iters, choice, colchoice, noise, aa)
        except ValueError:
            print("Input numbers only", end = '')
            sleep(1)
    tty.setraw(sys.stdin)

def mandelimggen(cx, cy, w, maxiters, res, number, noise, aa):
    # used to make the images when the user requests it can't be jit compiled because of PIL
    h = (float(w) * float(res[1])) / float(res[0])
    minx = cx - .5 * w
    maxx = cx + .5 * w
    miny = cy - .5 * h
    maxy = cy + .5 * h
    # generates the values for iterations per pixel
    iters = mandelimgfast(minx, maxx, miny, maxy, maxiters * 2, res[0] * aa, res[1] * aa)
    iters %= maxiters * 2
    maximum = iters.max()
    # making the list of colors for faster access while making image
    colors = cols(number, maximum + 1, noise, iters[iters != 0].min())

    image = itertoimage(iters, colors)

    img = Image.fromarray(image, "RGB")

    img.resize((res[0], res[1]), Image.LANCZOS).save(str('./images/' + str((ndec(cx), ndec(cy))) + ' ' + str(ndec(w)) + ' ' + str(res) + '.png'))
    img.resize((res[0], res[1]), Image.LANCZOS).show()

def juliaimggen(cx, cy, mx, my, w, maxiters, res, number, noise, aa):
    # used to make the images when the user requests it can't be jit compiled because of PIL
    h = (float(w) * float(res[1])) / float(res[0])
    minx = cx - .5 * w
    maxx = cx + .5 * w
    miny = cy - .5 * h
    maxy = cy + .5 * h
    # generates the values for iterations per pixel
    iters = juliaimgfast(minx, maxx, miny, maxy, mx, my, maxiters * 2, res[0] * aa, res[1] * aa)
    iters %= maxiters * 2
    maximum = iters.max()
    # making the list of colors for faster access while making image
    colors = cols(number, maximum + 1, noise, iters[iters != 0].min())

    image = itertoimage(iters, colors)

    img = Image.fromarray(image, "RGB")

    img.resize((res[0], res[1]), Image.LANCZOS).save(str('./images/' + str((ndec(cx), ndec(cy))) +str((ndec(mx), ndec(my))) + ' ' + str(ndec(w)) + ' ' + str(res) + '.png'))
    img.resize((res[0], res[1]), Image.LANCZOS).show()

@jit(parallel=True, nogil=True)
def mandelimgfast(minx, maxx, miny, maxy, maxiter, width, height):
    # different than the other mandelfast as this one just provides the pixel values which get cleaned by antialiasing later
    iters = empty((width, height), uint16)
    xspace = linspace(minx, maxx, width, dtype=float64)
    yspace = linspace(miny, maxy, height, dtype=float64)
    for x in tqdm(range(width)):
        for y in range(height):
            iters[x, y] = mandelfact(xspace[x], yspace[y], maxiter)
    return iters % maxiter

@jit(parallel=True, nogil=True)
def juliaimgfast(minx, maxx, miny, maxy, mx, my, maxiter, width, height):
    # different than the other mandelfast as this one just provides the pixel values which get cleaned by antialiasing later
    iters = empty((width, height), uint16)
    xspace = linspace(minx, maxx, width, dtype=float64)
    yspace = linspace(miny, maxy, height, dtype=float64)
    for x in tqdm(range(width)):
        for y in range(height):
            iters[x, y] = juliafact(xspace[x], yspace[y], mx, my, maxiter)
    return iters

def colorlerp(c1, c2, t): return (int(c1[0] + (c2[0] - c1[0]) * t), int(c1[1] + (c2[1] - c1[1]) * t), int(c1[2] + (c2[2] - c1[2]) * t)) # might be used in the future, will allow for cleaner gradients in picture outputs

@jit(parallel=True, nogil=True)
def colmap(c): return int(remap(c, 0, 1, 0, 255)) # just used to swap gradient from decimals to regular color values

ndec = lambda x, n = 3: 0 if x == 0 else round(x, -int(floor(log10(abs(x)))) + (n - 1)) # used to make the names of files cleaner and to make readout shorter in command line

def mandelzoom(cx, cy, w, maxiters, res, number, choice, noise):
    # used to make the images when the user requests it can't be jit compiled because of PIL
    h = (float(w) * float(res[1])) / float(res[0])
    minx = cx - .5 * w
    maxx = cx + .5 * w
    miny = cy - .5 * h
    maxy = cy + .5 * h
    # generates the values for iterations per pixel
    iters = mandelimgfast(minx, maxx, miny, maxy, maxiters, res[0], res[1])
    maximum = iters.max()
    # making the list of colors for faster access while making image
    colors = cols(choice, int(maximum + 1), noise, iters[iters != 0].min())

    image = itertoimage(iters, colors)

    img = Image.fromarray(image, "RGB")

    img.save(str('./zoomcache/' + str(number) + '.png'))

def mandeliter(cx, cy, w, maxiters, res, choice, noise, mode):
    # used to make the images when the user requests it can't be jit compiled because of PIL
    h = (float(w) * float(res[1])) / float(res[0])
    minx = cx - .5 * w
    maxx = cx + .5 * w
    miny = cy - .5 * h
    maxy = cy + .5 * h
    # generates the values for iterations per pixel
    iters = mandelimgfast(minx, maxx, miny, maxy, maxiters, res[0] * 2, res[1] * 2)
    maximum = int(percentile(iters[iters != 0], 99.9)) + 1
    minimum = iters[iters != 0].min()
    # making the list of colors for faster access while making image
    if mode == "Add":
        colors = cols(choice, int(maximum + 1), noise, minimum)
        for number in tqdm(range(maximum - minimum)):
            c = minimum + number
            image = itertoimage(clip(iters, 0, c) % c, colors)
            img = Image.fromarray(image, "RGB")
            img.resize((res[0], res[1]), Image.LANCZOS).save(str('./itercache/' + str(number) + '.png'))
    elif mode == "Single":
        for number in tqdm(range(maximum - minimum)):
            c = minimum + number
            colors = cols(choice, int(iters.max() + 1), noise, minimum, c)
            image = itertoimage(iters, colors)
            img = Image.fromarray(image, "RGB")
            img.resize((res[0], res[1]), Image.LANCZOS).save(str('./itercache/' + str(number) + '.png'))

    return maximum - minimum

@jit
def itertoimage(iters, colors):
    res = iters.shape
    image = empty((res[1], res[0], 3), uint8)
    # iters = clip(iters, 0, clipping) % clipping
    for x in tqdm(range(res[0])):
        for y in range(res[1]):
            image[y, x] = colors[int(iters[x, y])]
    return image

def juliaanimimage(mx, my, res, number, choice, noise):
    # used to make the images when the user requests it can't be jit compiled because of PIL
    h = (float(4) * float(res[1])) / float(res[0])
    minx = -2.0
    maxx = 2.0
    miny = -.5 * h
    maxy = .5 * h
    # generates the values for iterations per pixel
    iters = juliaimgfast(minx, maxx, miny, maxy, mx, my, 5000, res[0], res[1])
    iters %= 5000
    maximum = iters.max()
    # making the list of colors for faster access while making image
    colors = cols(choice, maximum + 1, noise, iters[iters != 0].min())

    image = itertoimage(iters, colors)

    img = Image.fromarray(image, "RGB")
    
    img.save(str('./janimcache/' + str(number) + '.png'))

def mandelanimrender(cx, cy, endw, iters, choice, noise):
    print(u'\u001b[0m\u001b[1000D\u001b[1000A')
    title = "Pick a type of animation:"
    options = ["Zoom", "Iterations", "Cancel"]
    animtype = pick(options, title)[0]
    if animtype == "Cancel":
        tty.setraw(sys.stdin)
        return

    title = "Pick a resolution for the rendered animation:"
    options = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), 'Cancel']
    resolution = pick(options, title)[0]
    if resolution == "Cancel":
        tty.setraw(sys.stdin)
        return
    system('reset')

    if animtype == "Zoom":
        frames = int(raw_input("Number of frames for final animation to have: "))
        iterkey = linspace(200, iters, num = frames, dtype=uint16)
        keysw = logspace(log10(4), log10(endw), frames, endpoint = True, base = 10.0)
        for i in tqdm(range(frames)):
            mandelzoom(cx, cy, keysw[i], iterkey[i], resolution, i, choice, noise)
        animname = str('./anim/' + str((ndec(cx), ndec(cy))) + ' ' + str(ndec(endw)) + ' ' + str(resolution) + '.mp4')
        with get_writer(animname, mode='I', fps = 60) as writer:
            for i in range(frames):
                image = imread('./zoomcache/' + str(i) + '.png')
                writer.append_data(image)
        for i in range(frames):
            remove('./zoomcache/' + str(i) + '.png')
    elif animtype == "Iterations":
        mode = pick(["Add", "Single"], "Choose Iteration Mode:")[0]
        frames = mandeliter(cx, cy, endw, iters, resolution, choice, noise, mode)
        animname = str('./anim/' + str((ndec(cx), ndec(cy))) + ' ' + str(iters) + ' ' + str(resolution) + '.mp4')
        with get_writer(animname, mode='I', fps = 30) as writer:
            for i in range(frames):
                image = imread('./itercache/' + str(i) + '.png')
                writer.append_data(image)
        for i in range(frames):
            remove('./itercache/' + str(i) + '.png')
        
    tty.setraw(sys.stdin)

def juliapan(waypoints, choice, noise):
    print(u'\u001b[0m\u001b[1000D\u001b[1000A')
    title = "Pick a resolution for the rendered animation:"
    options = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), 'Cancel']
    resolution = pick(options, title)[0]

    if resolution == "Cancel":
        tty.setraw(sys.stdin)
        return
    system('reset')
    frames = int(raw_input("Number of frames for final animation to have: "))
    mx = linspace(waypoints[0][0], waypoints[1][0], num=frames)
    my = linspace(waypoints[0][1], waypoints[1][1], num=frames)
    for i in tqdm(range(frames)):
        juliaanimimage(mx[i], my[i], resolution, i, choice, noise)
    animname = str('./anim/' + str((ndec(waypoints[0][0]), ndec(waypoints[0][1]))) + ' ' + str((ndec(waypoints[1][0]), ndec(waypoints[1][1]))) + ' ' + str(resolution) + '.mp4')
    with get_writer(animname, mode='I', fps = 60) as writer:
        for i in range(frames):
            image = imread('./janimcache/' + str(i) + '.png')
            writer.append_data(image)
    for i in range(frames):
        remove('./janimcache/' + str(i) + '.png')
    tty.setraw(sys.stdin)
    waypoints = []

def bookmarks(x, y, w, iters):
    bookmarks = loads(open("Bookmarks.json").read())
    marks = list(bookmarks)
    marks.append("+ New")
    marks.append("Cancel")
    location = pick(marks, "Choose a bookmark or make a new one")
    if location[0] == "+ New":
        system("reset")
        Namechoice = str(raw_input("Choose a name for the bookmark: "))
        bookmarks[Namechoice] = [x, y, w, iters]
        open("Bookmarks.json", "w").write(str(dumps(bookmarks, sort_keys = True, indent = 4)).replace("\'", '\"').replace("u", ""))
        return x, y, w, iters
    elif location[0] == "Cancel":
        return x, y, w, iters
    else:
        return bookmarks[location[0]]

loc = path.dirname(path.abspath(__file__))

def julialoop(mandelx, mandely, choice, noise):
    '''main loop for the command line drawing'''
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
        print(coords, end='')

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
        elif key.lower() == 'n': noise.newcolors()
        elif key.lower() == 'r': cx = 0; cy = 0; w = 4.0; maxiters = 100  			# resetting to default values
        elif key.lower() == 'f': juliaimg(cx, cy, mandelx, mandely, w, maxiters, choice, noise)
        elif key.lower() == ' ': break
        else:
            pass																	# if not an accepted input continues loop
        sys.stdout.flush()

if __name__ == '__main__':
    if not path.isdir("./images/"): mkdir("./images/")
    if not path.isdir("./zoomcache/"): mkdir("./zoomcache/")
    if not path.isdir("./itercache/"): mkdir("./itercache/")
    if not path.isdir("./janimcache/"): mkdir("./janimcache/")
    if not path.isdir("./anim/"): mkdir("./anim/")
    mandelloop()
