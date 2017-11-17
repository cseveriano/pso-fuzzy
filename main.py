import PSO
import math
import random as rnd
import numpy as np

def sinc(xx):
    return math.sin(xx)/xx

def createParticleVectors(nrules, nparams):
    vec = [[0] * nparams] * nrules
    return vec
#--- RUN ----------------------------------------------------------------------+

def initParticleValuesForSinc(v, nrules):
    for i in range(nrules):
        v[i][0] = rnd.random() * 2 * math.pi
        v[i][1] = rnd.random()
        v[i][2] = rnd.random() * (-1) ^ (math.round(rnd.random()))
        v[i][3] = rnd.random() * (-1) ^ (math.round(rnd.random()))
    return v

nrules = 8
nparams = 4

bounds = [10,10]
x = np.arange(0,2 * math.pi,0.01)
y = [sinc(xx) for xx in x]

initial = initParticleValuesForSinc(createParticleVectors(nrules, nparams), nrules)
PSO(x,y,initial,bounds,num_particles=100,maxiter=1000)
