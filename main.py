import PSO
import math
import numpy as np

def sinc(xx):
    return math.sin(xx)/xx


nrules = 8
nparams = 4

bounds = [10,10]
x = np.arange(0.01, 2 * math.pi, 0.01)
y = [sinc(xx) for xx in x]


PSO.PSO(x,y,nrules, nparams,bounds,num_particles=100,maxiter=100)
