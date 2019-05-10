import numpy as np
from pyswarms.single.global_best import GlobalBestPSO

# 粒子群优化算法
class MPSO(object):
    def __init__(self,valueRange):

        x_min = valueRange[:,0]
        x_max = valueRange[:,1]
        self.xRange = (x_min, x_max)

    def train(self,calFun, kwargs):

        options = {'c1': 1.5, 'c2': 1.3, 'w': 0.9}
        bounds = self.xRange
        dim = len(self.xRange[0])
        optimizer = GlobalBestPSO(n_particles=20, dimensions=dim, options=options, bounds=bounds)
        cost, pos = optimizer.optimize(calFun, 25, **kwargs)
        return pos,cost




class PSO(object):
    def __init__(self, valueRange, opt = 1):

        self.opt = opt
        self.Wini = 0.9
        self.Wend = 0.4

        self.c1 = 1.5
        self.c2 = 1.3

        self.maxgen = 25
        self.sizepop = 20

        self.posMax = valueRange[:, 1]
        self.posMin = valueRange[:, 0]

        self.Vmax = 0.1 * self.posMax
        self.Vmin = -0.1 * self.posMax

        num = valueRange.shape[0]
        self.num = num
        self.pop = np.zeros((self.sizepop, num))
        self.v = np.zeros((self.sizepop, num))

        for i in range(num):
            self.pop[:, i] = np.random.uniform(self.posMin[i], self.posMax[i], (self.sizepop, ))
            self.v[:,i] = np.random.uniform(self.Vmin[i], self.Vmax[i], (self.sizepop, ))


    def train(self,Func,kwargs):


        kwargs['pop'] = self.pop
        fitness = Func(**kwargs)

        if self.opt==1:
            i = np.argmin(fitness)
        else:
            i = np.argmax(fitness)

        gbest = self.pop
        zbest = self.pop[i,:]
        fitnessgbest = fitness
        fitnesszbest = fitness[i]
        t = 0
        record = np.zeros(self.maxgen)
        Vmaxtile = np.tile(self.Vmax.reshape(1, -1), (self.sizepop, 1))
        Vmintile = np.tile(self.Vmin.reshape(1, -1), (self.sizepop, 1))
        posMaxtile = np.tile(self.posMax.reshape(1, -1), (self.sizepop, 1))
        posMintile = np.tile(self.posMin.reshape(1, -1), (self.sizepop, 1))

        while t < self.maxgen:
            # 速度更新
            W = self.Wini - (self.Wini - self.Wend)*t/self.maxgen
            self.v = W * self.v + self.c1 * np.random.random()*(gbest - self.pop) + self.c2 * np.random.random()*(zbest.reshape(1, -1) - self.pop)
            self.v[self.v > Vmaxtile] = Vmaxtile[self.v > Vmaxtile]
            self.v[self.v < Vmintile] = Vmintile[self.v < Vmintile]

            # 种群更新
            self.pop = self.pop + self.v
            self.pop[self.pop > posMaxtile] = posMaxtile[self.pop > posMaxtile]
            self.pop[self.pop < posMintile] = posMintile[self.pop < posMintile]


            # 自适应变异
            p = np.random.random()
            if p > 0.4:
                k = np.random.randint(0, self.sizepop-1)
                for L in range(self.num):
                    self.pop[k, L] = np.random.uniform(self.posMin[L], self.posMax[L])

            kwargs['pop'] = self.pop
            fitness = Func(**kwargs)
            # 个体最优位置更新
            if self.opt == 1:
                index = fitness < fitnessgbest
            else:
                index = fitness > fitnessgbest

            fitnessgbest[index] = fitness[index]
            gbest[index, :] = self.pop[index, :]

            # 群体最优更新
            j = np.argmin(fitness)
            if fitness[j] < fitnesszbest:
                zbest = self.pop[j, :]
                fitnesszbest = fitness[j]

            if self.opt==1:
                j = np.argmin(fitness)
                if fitness[j] < fitnesszbest:
                    zbest = self.pop[j, :]
                    fitnesszbest = fitness[j]
            else:
                j = np.argmax(fitness)
                if fitness[j] > fitnesszbest:
                    zbest = self.pop[j, :]
                    fitnesszbest = fitness[j]


            record[t] = fitnesszbest  # 记录群体最优位置的变化
            print('MSECV[%d]: %.3f' % (t,record[t]))
            t = t + 1

        return zbest,record









