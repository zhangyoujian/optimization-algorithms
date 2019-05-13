import numpy as np

class PSO(object):
    def __init__(self,valueRange,opt=1,popSize=20, epoch=200):

        self.opt = opt
        self.Wini = 0.9
        self.Wend = 0.4

        self.c1 = 1.5
        self.c2 = 1.3

        self.maxgen = epoch
        self.sizepop = popSize

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

        print('开始粒子群优化')
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
            if p > 0.5:
                k = np.random.randint(0, self.sizepop)
                var = np.random.randint(0, self.num)
                self.pop[k, var] = np.random.uniform(self.posMin[var], self.posMax[var])
                # for L in range(self.num):
                #     self.pop[k, L] = np.random.uniform(self.posMin[L], self.posMax[L])

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
            # print('MSECV[%d]: %.3f' % (t,record[t]))
            print("%d/%d[==============]Loss:%.4f" % (t + 1, self.maxgen, record[t]))
            t = t + 1

        return zbest,record









