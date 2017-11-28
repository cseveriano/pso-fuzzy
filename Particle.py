import random
import math

class Particle:
    def __init__(self,x0):
        self.velocity=[]          # particle velocity
        self.pos_best=[]          # best position individual
        self.error = -1           # individual error
        self.err_best = -1        # best individual error
        self.num_parameters = len(x0[0])
        self.num_rules = len(x0)
        self.position = x0
        self.velocity = [[random.uniform(-1,1)] * self.num_parameters] * self.num_rules

    def evaluate(self, x, y):

        y_est = self.yEstimated(x,y)

        diff = [(y1-y2) ** 2 for y1,y2 in zip(y,y_est)]
        rmse = math.sqrt(sum(diff) / len(x))
        self.error = rmse

        if (self.error < self.err_best) or (self.err_best == -1):
            self.pos_best = self.position
            self.err_best = self.error

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.6       # constant inertia weight (how much to weigh the previous velocity)
        c1=1.49445        # cognative constant
        c2=1.49445        # social constant

        for r in range(self.num_rules):
            for i in range(self.num_parameters):
                r1=random.random()
                r2=random.random()

                vel_cognitive = c1 * r1 * (self.pos_best[r][i] - self.position[r][i])
                vel_social = c2 * r2 * (pos_best_g[r][i] - self.position[r][i])
                self.velocity[r][i] = (w*self.velocity[r][i]) + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for r in range(self.num_rules):
            for i in range(self.num_parameters):
                self.position[r][i]=self.position[r][i]+self.velocity[r][i]

            # adjust maximum position if necessary
            #if self.position[i]>bounds[i][1]:
            #    self.position[i]=bounds[i][1]

            # adjust minimum position if neseccary
            #if self.position[i] < bounds[i][0]:
            #    self.position[i]=bounds[i][0]

    def gaussianMembership(self, xx, med, std):
        return math.exp(-0.5 * ((xx - med) / std) ** 2)


    def yEstimated(self, x, y):

        y_est = []
        for xx in x:
            num = 0
            den = 0
            for r in range(self.num_rules):
                wk = self.gaussianMembership(xx, self.position[r][0], self.position[r][1])
                yk = self.position[r][2] * xx + self.position[r][3]
                num += wk * yk
                #den += wk

            #y_est.append(num / den)
            y_est.append(num)

        return y_est