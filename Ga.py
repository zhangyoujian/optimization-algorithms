# -*- coding: utf-8 -*-
import numpy as np
import math
import random
import copy

def calcbits(bounds,precision):
    xRange = bounds[:,1] - bounds[:,0]
    bits = np.ceil(np.log2(xRange/precision))
    return bits

class Ga(object):
    def __init__(self, popSize, bounds, pc, pm, options=(1e-6, 1)):
        self.popSize = popSize
        self.bounds = bounds
        self.pc = pc
        self.pm = pm
        self.options = options
        self.maxGen = 500
        self.numVar = bounds.shape[0]

    def initPop(self):
        popSize = self.popSize
        bounds = self.bounds
        chromLength = calcbits(bounds,self.options[0])
        bitLength = int(np.sum(chromLength))
        pop = np.round(np.random.random([popSize,bitLength]))
        return pop



    def selection(self,pop, fitvalue):

        [px, py] = pop.shape[:]
        fitvalue = fitvalue + np.min(fitvalue)
        totalfit = np.sum(fitvalue)
        p_fitvalue = fitvalue / totalfit
        p_fitvalue = np.cumsum(p_fitvalue)
        ms = np.sort(np.random.random(px))
        fitin = 0
        newin = 0
        newpop = None

        while newin < px:
            if ms[newin] < p_fitvalue[fitin]:
                if newpop is None:
                    newpop = pop[fitin, :].reshape((1, -1))
                else:
                    newpop = np.concatenate((newpop, pop[fitin, :].reshape((1, -1))), axis=0)
                newin += 1

            else:
                fitin += 1
        return newpop


    def crossover(self,pop):
        [px, py] = pop.shape[:]
        newpop = np.ones(pop.shape)

        for i in range(0,px-1,2):
            if random.random() < self.pc:
                cpoint =round(random.random()*py)
                if cpoint==py:
                    cpoint = py-1
                newpop[i,:] = np.concatenate((pop[i,0:cpoint],pop[i+1,cpoint:]),axis=0)
                newpop[i+1, :] = np.concatenate((pop[i+1,0:cpoint], pop[i, cpoint:]), axis=0)
            else:
                newpop[i, :] = pop[i,:]
                newpop[i+1, :] = pop[i+1, :]
        return newpop


    def mutation(self,pop):
        [px, py] = pop.shape[:]
        newpop = np.ones(pop.shape)

        for i in range(px):
            if random.random() <self.pm:
                mpoint = round(random.random()*py)
                if mpoint==py:
                    mpoint = py-1
                newpop[i,:] = pop[i,:]
                if newpop[i,mpoint]==0:
                    newpop[i, mpoint] = 1
                else:
                    newpop[i, mpoint] = 0
            else:
                newpop[i, :] = pop[i, :]
        return newpop

    @staticmethod
    def binary2decimal(pop,bounds,eps):

        numVar = bounds.shape[0]
        chromList = calcbits(bounds, eps)

        xrange = bounds[:,1] - bounds[:,0]
        [px, py] = pop.shape[:]
        startIndex = 0
        FinalPos = np.zeros((px,numVar))

        for k in range(numVar):
            singlePop = pop[:,startIndex:startIndex+int(chromList[k])]
            startIndex = startIndex + int(chromList[k])
            py = int(chromList[k])
            pop1 = None
            for i in range(py):
                temp = np.power(2, (py - 1 - i) * singlePop[:, i])
                if pop1 is None:
                    pop1 = temp.reshape((-1, 1))
                else:
                    pop1 = np.concatenate((pop1, temp.reshape((-1, 1))), axis=1)
            temp = np.sum(pop1, axis=1)
            FinalPos[:, k] = temp * xrange[k]/(np.power(2.0, chromList[k]) - 1) + bounds[k,0]

        return FinalPos


    def train(self,Func,kwargs):

        pop = self.initPop()
        decimalValue = Ga.binary2decimal(pop,self.bounds,self.options[0])
        kwargs['pop'] = decimalValue
        fitness = Func(**kwargs)

        if self.options[1]==1:
            index = np.argmin(fitness)
            bestfit = np.min(fitness)
        else:
            index = np.argmax(fitness)
            bestfit = np.max(fitness)

        bestSolution = pop[index,:]
        record = np.zeros(self.maxGen)
        for i in range(self.maxGen):

            # 选择操作
            newpop = self.selection(pop,fitness)

            #交叉操作
            newpop = self.crossover(newpop)

            #变异操作
            newpop = self.mutation(newpop)

            decimalValue = Ga.binary2decimal(newpop, self.bounds, self.options[0])
            kwargs['pop'] = decimalValue
            fitness = Func(**kwargs)

            if self.options[1] == 1:
                index = np.argmin(fitness)
                bestFitValue = np.min(fitness)
            else:
                index = np.argmax(fitness)
                bestFitValue = np.max(fitness)

            bestindividual = newpop[index, :]

            if self.options[1] == 1:
                if bestFitValue < bestfit:
                    bestfit = bestFitValue
                    bestSolution = bestindividual
            else:
                if bestFitValue > bestfit:
                    bestfit = bestFitValue
                    bestSolution = bestindividual

            record[i] = bestfit
            print('Fitness[%d]: %.3f' % (i, record[i]))
            pop = newpop

        bestSolution = bestSolution.reshape(1,-1)
        bestSolution = Ga.binary2decimal(bestSolution, self.bounds, self.options[0])


        bestSolution = bestSolution.reshape(bestSolution.shape[1],)
        return bestSolution, record






