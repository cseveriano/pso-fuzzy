from Particle import Particle
import random as rnd
import math
import matplotlib.pyplot as plt

class PSO():
    def __init__(self,x,y,nrules, nparams,bounds,num_particles,maxiter):

        err_best_g=-1                 # best error
        g_best = []                   # best particle

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(self.initParticleValuesForSinc(self.createParticleVectors(nrules, nparams), nrules)))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(x,y)

                # determine if current particle is the best (globally)
                if swarm[j].error < err_best_g or err_best_g == -1:
                    g_best = swarm[j]
                    err_best_g = float(swarm[j].error)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(g_best.position)
                swarm[j].update_position(bounds)

            print("Iteracao ",i,"\n")
            print("Erro RMSE: ",err_best_g,"\n")
            i+=1

        # print final results
        print('ERRO FINAL:')
        print(err_best_g)
        y_final_est = g_best.yEstimated(x,y)
        plt.plot(x, y)
        plt.plot(x, y_final_est)

    def createParticleVectors(self, nrules, nparams):
        vec = [[0] * nparams for n in range(nrules)]
        return vec
#--- RUN ----------------------------------------------------------------------+

    def initParticleValuesForSinc(self, v, nrules):
        for i in range(nrules):
            v[i][0] = rnd.random() * 2 * math.pi
            v[i][1] = rnd.random()
            v[i][2] = rnd.random() * (-1) ** (round(rnd.random()))
            v[i][3] = rnd.random() * (-1) ** (round(rnd.random()))
        return v
