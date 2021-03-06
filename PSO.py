from Particle import Particle
import random as rnd
import math
import matplotlib.pyplot as plt

class PSO():
    def __init__(self,x,nrules, num_particles,maxiter):

        self.err_best_g=-1                 # best error
        self.g_best = []                   # best particle
        self.maxiter = maxiter
        # establish the swarm
        self.swarm = []
        for i in range(0,num_particles):
            self.swarm.append(Particle(self.initParticleValuesForHousePrice(self.createParticleVectors(nrules,x), nrules, x)))

    def train(self,x,y):
        num_particles = len(self.swarm)
        # begin optimization loop
        i=0
        while i < self.maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                self.swarm[j].evaluate(x,y)

                # determine if current particle is the best (globally)
                if self.swarm[j].error < self.err_best_g or self.err_best_g == -1:
                    self.g_best = self.swarm[j]
                    self.err_best_g = float(self.swarm[j].error)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                self.swarm[j].update_velocity(self.g_best.position)
                self.swarm[j].update_position()

            print("Iteracao ",i,"\n")
            print("Erro RMSE: ",self.err_best_g,"\n")
            i+=1

        # print final results
        print('ERRO FINAL:')
        print(self.err_best_g)
        y_final_est = self.g_best.yEstimated(x)
        plt.plot(x, y)
        plt.plot(x, y_final_est)
        return self.g_best

    def createParticleVectors(self, nrules, x):
        # modelo usa pertinencia gaussiana
        # portanto, temos media, variancia, alfa de cada atributo e alfa do b (modelo linear)
        # numero de parametros é (3*n)+1
        nparams = (3 * len(x.columns)) + 1
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

    # TODO: Criar uma subclasse de House Price
    def initParticleValuesForHousePrice(self, v, nrules, x):
        # FORMATO DA REGRA Ri:
        # med_i1,var_i1, med_i2,var_i2, ..., med_in,var_in,alfa_i1,alfa_i2,...,alfa_in,alfa_i0
        n = len(x.columns)

        for i in range(nrules):
            for j in range(n):
                v[i][2*j] = rnd.uniform(x[x.columns[j]].min(), x[x.columns[j]].max()) # media
                v[i][(2*j)+1] = rnd.uniform(0,1) # variancia

            for j in range(n):
                v[i][(2*n) + j] = rnd.uniform(-1,1) # alfa

            v[i][(3*n)] = rnd.uniform(-1,1) # alfa_0
        return v
