'''
Created on May 7, 2015

@author: wohlhart
'''
import cPickle
import matplotlib.pyplot as plt
import sys
import numpy


def loadTrainResults(fileName):
    with open(fileName, 'rb') as f: 
        descrNet = cPickle.load(f)
        trainProgressData = cPickle.load(f)
#        wvals = cPickle.load(f)
        cfg = cPickle.load(f)   
        trainParams = cPickle.load(f)
#         imgsPklFileName = cPickle.load(f)
#         trainSetPklFileName = cPickle.load(f) 
    return trainProgressData

if __name__ == '__main__':
    
    
    fileName = sys.argv[1]
    trainProgressData = loadTrainResults(fileName)
    
    train_costs = trainProgressData[0]['train_costs']
    for i in xrange(1,len(trainProgressData)):
        x = trainProgressData[i]['train_costs']
        train_costs.extend(x)
    print(numpy.max(train_costs))
    
    plt.plot(train_costs)
    plt.ylim((0,numpy.max(train_costs)))
    
    plt.show()
    
    