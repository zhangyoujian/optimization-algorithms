import numpy as np


class Ant(object):
    def __init__(self,xrange,PopSize=100,opt=1,epoch=50):
        self.AntSize = PopSize
        self.opt = opt
        self.ECHO = epoch
        self.xrange = xrange
        self.numVar = xrange.shape[0]
        self.pop = np.zeros((PopSize, self.numVar))
        for i in range(self.numVar):
            self.pop[:, i] = np.random.uniform(xrange[i,  0], xrange[i, 1], PopSize)


    def train(self,Func, kwargs):

        print('开始蚁群优化')
        # 初始化信息素
        kwargs['pop'] = self.pop
        eps = 1e-4
        T0 = Func(**kwargs)
        if self.opt == 1:
            T0 = 1/(T0 + eps)

        # 开始寻优
        P0 = 0.2       #全局转移概率
        P = 0.8        #信息素蒸发系数

        T_Best = np.zeros(self.ECHO)
        Prob = np.zeros((self.ECHO, self.AntSize))
        max_local = np.zeros(self.ECHO)
        max_global = np.zeros(self.ECHO)

        for Echo in range(self.ECHO):
            lambdaStep = 1/(Echo+1)      #转移步长参数
            T_Best[Echo] = np.max(T0)
            BestIndex = np.argmax(T0)

            r = T0[BestIndex] - T0
            Prob[Echo, :] = r/T0[BestIndex]

            for j_g_tr in range(self.AntSize):
                if Prob[Echo, j_g_tr] < P0:    #局部寻优
                   temp = self.pop[j_g_tr, :] + (2*np.random.rand(self.numVar)-1)*lambdaStep
                else: #全局寻优
                    temp = self.pop[j_g_tr, :] + (self.xrange[:, 1] - self.xrange[:, 0]) * (np.random.rand(self.numVar)- 0.5)*lambdaStep


                temp[temp<self.xrange[:,0]] = self.xrange[:,0][temp<self.xrange[:,0]]
                temp[temp > self.xrange[:, 1]] = self.xrange[:,1][temp > self.xrange[:, 1]]

                temp = temp.reshape((1,-1))
                kwargs['pop'] = temp
                F1 = Func(**kwargs)
                if self.opt == 1:
                    F1 = 1 / (F1 + eps)

                kwargs['pop'] = self.pop[j_g_tr, :].reshape((1,-1))
                F2 = Func(**kwargs)
                if self.opt == 1:
                    F2 = 1 / (F2 + eps)

                if F1 > F2:
                    self.pop[j_g_tr, :] = temp

            # 信息素更新

            kwargs['pop'] = self.pop
            F = Func(**kwargs)
            if self.opt == 1:
                F = 1 / (F + eps)
            T0 = (1 - P) * T0 + F

            i_iter = np.argmax(T0)
            max_local[Echo] = F[i_iter]
            if Echo>=1:
                if max_local[Echo]>=max_global[Echo]:
                    max_global[Echo] = max_local[Echo]
                    bestSolution = self.pop[i_iter,:]
                else:
                    max_global[Echo] =  max_global[Echo-1]
            else:
                max_global[Echo] = max_local[Echo]
                bestSolution = self.pop[i_iter, :]

            print("%d/%d[==============]Loss:%.4f"%(Echo+1,self.ECHO,max_global[Echo]))

        return bestSolution, max_global




