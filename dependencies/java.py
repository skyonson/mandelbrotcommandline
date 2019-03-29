from numba import jit
@jit
def remap(value, origlow, orighigh, newlow, newhigh):
    '''returns the remapped value'''
    oldrange = (float(orighigh) - float(origlow))  
    newrange = (float(newhigh) - float(newlow))  
    newvalue = float((((float(value) - float(origlow)) * float(newrange)) / float(oldrange)) + float(newlow))
    return float(newvalue)
