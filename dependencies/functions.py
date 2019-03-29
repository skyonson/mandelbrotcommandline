from __future__ import print_function

import sys
import tty
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
from colors import noiseColor, cols


@jit(parallel=True, nopython=True, nogil=True)
def juliafact(x, y, mx, my, iterations):
    '''does julia factorization for a given point up to a certain number of iterations'''
    # standard julia set function
    n = 1
    while n < iterations and x * x + y * y < 4.0:
        aa = x * x - y * y
        bb = 2 * x * y
        x = aa + mx
        y = bb + my
        n += 1
    return n

@jit(parallel=True, nogil=True)
def juliafast(minx, maxx, miny, maxy, mandelx, mandely, maxiter, width, height):
    '''returns an \'antialiased\' list of iters for the julia set'''
    # initializing arrays for:
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

@jit(parallel=True, nopython=True, nogil=True)
def mandelfact(x, y, iterations):
    '''does mandelbrot factorization for a given point up to a certain number of iterations'''
    # optimization to avoid calculations for the main and secondary bulbs
    q = ((x - .25) ** 2) + (y ** 2)
    if q * (q + x - .25) < .25 * y ** 2:
        return iterations
    if (x + 1) ** 2 + y ** 2 <= 1/16.0:
        return iterations
    # basic mandelbrot function implementation no optimizations other than putting the exponent check in the while loop to avoid an extra if
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

@jit(parallel=True, nogil=True)
def mandelfast(minx, maxx, miny, maxy, maxiter, width, height):
    '''returns an \'antialiased\' list of iters for the mandelbrot set'''
    # initializing arrays for:
    iterxl = empty((width * 2, height * 2), uint16)  # larger iteration array
    iters = empty((width, height), uint16)			# final iteration array
    xspace = linspace(minx, maxx, width * 2, dtype=float64)		# faster to access xcoords
    yspace = linspace(miny, maxy, height * 2, dtype=float64)		# faster to access ycoords

    # generating initial iterations
    for x in range(width * 2):
        for y in range(height * 2):
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

def mandelimg(cx, cy, w, iters, colchoice, noise):
    '''used to call mandelimggen with all variables, also provides selection to choose resolution'''
    system('reset')
    title = "Pick a resolution for the rendered image:"
    options = [(1920, 1080), (2560, 1440), (3840, 2160),
               (7680, 4320), 'Custom', 'Cancel']            # gives list of standard 16:9 resolutions
    choice = pick(options, title)[0]
    if choice == 'Custom':
        try:
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
            aa = int(raw_input("AA Samples: "))
            mandelimggen(cx, cy, w, iters, choice, colchoice, noise, aa)
        except ValueError:
            print("Input numbers only", end = '')
            sleep(1)
    tty.setraw(sys.stdin)

def mandelimggen(cx, cy, w, maxiters, res, number, noise, aa):
    '''used to make the images when the user requests it can't be jit compiled because of PIL'''
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

    img.resize((res[0], res[1]), Image.LANCZOS).save(str('./images/' + str((ndec(cx), ndec(cy))) + ' ' + str(ndec(w)) + ' ' + str(res) + '.png')) # saves image as coords, width and iters
    img.resize((res[0], res[1]), Image.LANCZOS).show()

@jit(parallel=True, nogil=True)
def mandelimgfast(minx, maxx, miny, maxy, maxiter, width, height):
    '''different than the other mandelfast as this one just provides the pixel values which get cleaned by antialiasing later'''
    iters = empty((width, height), uint16) # makes empty array to populate with iterations
    xspace = linspace(minx, maxx, width, dtype=float64)     # creates real values for quick access
    yspace = linspace(miny, maxy, height, dtype=float64)    # creates imaginary values for quick access
    for x in tqdm(range(width)):
        for y in range(height):
            iters[x, y] = mandelfact(xspace[x], yspace[y], maxiter)
    return iters % maxiter # sets maxiters to 0 for quicker coloration of max vals

def mandelzoom(cx, cy, w, maxiters, res, number, choice, noise):
    '''generates image for zoom animation'''
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
    '''generates images for iteration animation'''
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

def mandelanimrender(cx, cy, endw, iters, choice, noise):
    '''has user choose between animation types for the mandelbrot set'''
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

def juliaimg(cx, cy, mx, my, w, iters, colchoice, noise):
    '''used to call mandelimggen with all variables, also provides selection to choose resolution'''
    print(u'\u001b[0m' + u'\u001b[1000D' + u'\u001b[1000A')
    title = "Pick a resolution for the rendered image:"
    options = [(1920, 1080), (2560, 1440), (3840, 2160),
               (7680, 4320), 'Custom', 'Cancel']            # gives list of standard 16:9 resolutions
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

def juliaimggen(cx, cy, mx, my, w, maxiters, res, number, noise, aa):
    '''used to make the images when the user requests it can't be jit compiled because of PIL'''
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
def juliaimgfast(minx, maxx, miny, maxy, mx, my, maxiter, width, height):
    '''different than the other mandelfast as this one just provides the pixel values which get cleaned by antialiasing later'''
    iters = empty((width, height), uint16) # makes empty array to populate with iterations
    xspace = linspace(minx, maxx, width, dtype=float64)     # creates real values for quick access
    yspace = linspace(miny, maxy, height, dtype=float64)    # creates imaginary values for quick access
    for x in tqdm(range(width)):
        for y in range(height):
            iters[x, y] = juliafact(xspace[x], yspace[y], mx, my, maxiter)
    return iters

ndec = lambda x, n = 3: 0 if x == 0 else round(x, -int(floor(log10(abs(x)))) + (n - 1)) # used to make the names of files cleaner and to make readout shorter in command line
@jit
def itertoimage(iters, colors):
    '''makes array of colors from iterations to make the image with fromarray'''
    res = iters.shape
    image = empty((res[1], res[0], 3), uint8)
    # iters = clip(iters, 0, clipping) % clipping
    for x in tqdm(range(res[0])):
        for y in range(res[1]):
            image[y, x] = colors[int(iters[x, y])]
    return image

def juliaanimimage(mx, my, res, number, choice, noise):
    '''makes frames for panning across the julia set'''
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

def juliapan(waypoints, choice, noise):
    '''gives user choices for resolution for julia pan animation'''
    print(u'\u001b[0m\u001b[1000D\u001b[1000A')
    title = "Pick a resolution for the rendered animation:"
    options = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), 'Cancel']
    resolution = pick(options, title)[0]

    if resolution == "Cancel":
        tty.setraw(sys.stdin)
        return
    system('reset')
    frames = int(raw_input("Number of frames for final animation to have: "))
    mx = linspace(waypoints[0][0], waypoints[1][0], num=frames) # makes keyframes for animation to use
    my = linspace(waypoints[0][1], waypoints[1][1], num=frames) # makes keyframes for animation to use
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
    '''Lets user choose from an existing bookmark or make a new one'''
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

