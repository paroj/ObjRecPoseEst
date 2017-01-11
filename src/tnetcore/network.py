'''
Created on 08.05.2014

@author: pwohlhart
'''
from __future__ import print_function

import numpy
import time
import theano
import theano.tensor as T

from tnetcore.layers.convlayer import ConvLayerParams
from tnetcore.layers.hiddenlayer import HiddenLayerParams
from tnetcore.layers.logisticregression import LogisticRegressionParams
from tnetcore.layers.poollayer import PoolLayerParams
from tnetcore.layers.actflayer import ActivationFunctionLayerParams, ReLU
from tnetcore.data import DataManager
import pydot
import logging
import cPickle
import os


class NetworkParams(object):
    
    def __init__(self,typeID,nChan=3,wIn=64,hIn=64,batchSize=128,nOut=3):
        '''
        Init one of the parametrizations, depeding on type
        
        :type typeID: int
        :param typeID: type of descr network
        '''
        self.nwClass = Network
        
        self.batch_size = batchSize
        
        self.typeID = typeID

        self.layerCfgs = []
        self.outputDim = None 
        
#         if (typeID == 0):
#         
#             self.layer0 = ConvLayerParams(inputDim = (batchSize,nChan,64,64),  # w,h,nChannel
#                                           nFilters = 10,
#                                           filterDim = (9,9),  
#                                           poolsize = (2,2))
#             
#             self.layer1 = ConvLayerParams(inputDim = self.layer0.outputDim,
#                                           nFilters = 10,
#                                           filterDim = (5,5),
#                                           poolsize = (2,2))
#             
#             l1out = self.layer1.outputDim
#             self.layer2 = HiddenLayerParams(inputDim = (l1out[0],l1out[1]*l1out[2]*l1out[3]),  
#                                             outputDim = (batchSize,500),
#                                             activation = T.tanh)
#             
#             self.layer3 = HiddenLayerParams(inputDim = self.layer2.outputDim,  
#                                             outputDim = (batchSize,128),
#                                             activation = T.nnet.sigmoid)
#             self.outputDim = self.layer3.outputDim
#             self.layerCfgs = [self.layer0,self.layer1,self.layer2,self.layer3]
# 
#         elif (typeID == 1):     
#                
#             self.layer0 = ConvLayerParams(inputDim = (batchSize,nChan,64,64),  # w,h,nChannel
#                                           nFilters = 10,
#                                           filterDim = (9,9),  
#                                           poolsize = (2,2))
#             
#             self.layer1 = ConvLayerParams(inputDim = self.layer0.outputDim,
#                                           nFilters = 10,
#                                           filterDim = (5,5),
#                                           poolsize = (2,2))
#             
#             l1out = self.layer1.outputDim
#             self.layer2 = HiddenLayerParams(inputDim = (l1out[0],l1out[1]*l1out[2]*l1out[3]),  
#                                             outputDim = (batchSize,500),
#                                             activation = T.tanh)
#                            
#             self.layer3 = LogisticRegressionParams(inputDim = self.layer2.outputDim,
#                                                    outputDim = (batchSize,128))
#             self.outputDim = self.layer3.outputDim
#             self.layerCfgs = [self.layer0,self.layer1,self.layer2,self.layer3]
# 
#         elif (typeID == 2):
#                
#             self.layer0 = ConvLayerParams(inputDim = (batchSize,nChan,wIn,hIn),  # w,h,nChannel
#                                           nFilters = 5,#10,
#                                           #filterDim = (9,9),  
#                                           filterDim = (7,7),
#                                           poolsize = (2,2))
#             
#             self.layer1 = ConvLayerParams(inputDim = self.layer0.outputDim,
#                                           nFilters = 10,
#                                           filterDim = (5,5),
#                                           poolsize = (2,2))
#             
#             l1out = self.layer1.outputDim
#             self.layer2 = HiddenLayerParams(inputDim = (l1out[0],l1out[1]*l1out[2]*l1out[3]),  
#                                             #outputDim = (bs,500),
#                                             outputDim = (batchSize,3),
#                                             activation = T.nnet.sigmoid)
#             self.outputDim = self.layer2.outputDim
#             self.layerCfgs = [self.layer0,self.layer1,self.layer2]
# 
#         elif (typeID == 3):  # LeNet-5
#             
#             #activation = lambda
#             #activation = lambda x: x * (x > 0)
#             
#             self.layer0 = ConvLayerParams(inputDim = (batchSize,nChan,wIn,hIn),  # w,h,nChannel
#                                           nFilters = 12, #6,#10,
#                                           filterDim = (9,9), # (5,5),  
#                                           #poolType = 1,  TODO, mean pooling + sigmoid
#                                           poolsize = (2,2),
#                                           #activation = T.tanh)
#                                           activation = ReLU)
#             
#             self.layer1 = ConvLayerParams(inputDim = self.layer0.outputDim,
#                                           nFilters = 16,
#                                           filterDim = (5,5),
#                                           #poolType = 1,
#                                           poolsize = (2,2),
#                                           #activation = T.tanh)
#                                           activation = ReLU)
#             
#             l1out = self.layer1.outputDim
#             self.layer2 = HiddenLayerParams(inputDim = (l1out[0],l1out[1]*l1out[2]*l1out[3]),  
#                                             outputDim = (batchSize,120),
#                                             #activation = T.nnet.sigmoid),
#                                             activation = ReLU)
#             
#             self.layer3 = HiddenLayerParams(inputDim = self.layer2.outputDim,  
#                                             outputDim = (batchSize,84),
#                                             activation = T.nnet.sigmoid)  # last one stays sigmoid
# 
#             self.outputDim = self.layer3.outputDim
#             self.layerCfgs = [self.layer0,self.layer1,self.layer2,self.layer3]            
#         
#         if (typeID == 4):  # regression
#             self.layer0 = ConvLayerParams(inputDim = (batchSize,nChan,wIn,hIn),  # w,h,nChannel
#                                           nFilters = 16,#10,
#                                           #filterDim = (9,9),  
#                                           filterDim = (7,7),
#                                           poolsize = (2,2))
#             
#             self.layer1 = ConvLayerParams(inputDim = self.layer0.outputDim,
#                                           nFilters = 10,
#                                           filterDim = (5,5),
#                                           poolsize = (2,2))
#             
#             l1out = self.layer1.outputDim
#             self.layer2 = HiddenLayerParams(inputDim = (l1out[0],l1out[1]*l1out[2]*l1out[3]),  
#                                             #outputDim = (bs,500),
#                                             outputDim = (batchSize,20),
#                                             activation = ReLU)# relu
# 
#             self.layer3 = HiddenLayerParams(inputDim = self.layer2.outputDim,  
#                                             outputDim = (batchSize,1),  # regress 1 value
#                                             activation = None)  # linear output for the regression target
# 
#             self.outputDim = self.layer3.outputDim
#             self.layerCfgs = [self.layer0,self.layer1,self.layer2,self.layer3]
#             
#         if (typeID == 5):  # 
#             
#             self.layerCfgs = [] 
#             self.layer0 = ConvLayerParams(inputDim = (batchSize,nChan,wIn,hIn),  # w,h,nChannel
#                                           nFilters = 16,#10,
#                                           filterDim = (8,8),
#                                           wInitMode = 1,  # random init
#                                           #wInitMode = 2,  # sinusoids init
#                                           wInitOrthogonal = False)
#             self.layerCfgs.append(self.layer0)
#             
#             self.layer0pool = PoolLayerParams(inputDim = self.layer0.outputDim,
#                                               poolsize = (2,2))
#             self.layerCfgs.append(self.layer0pool)
#             
#             self.layer0actf = ActivationFunctionLayerParams(inputDim = self.layer0pool.outputDim,
#                                                             activation = ReLU)
#             self.layerCfgs.append(self.layer0actf)
#             
#             self.layer1 = ConvLayerParams(inputDim = self.layer0actf.outputDim,
#                                           nFilters = 7,
#                                           filterDim = (5,5),
#                                           wInitMode = 1, # random init
#                                           #wInitMode = 2, # sinusoids init
#                                           wInitOrthogonal = False)
#             self.layerCfgs.append(self.layer1)      
#             
#             self.layer1pool = PoolLayerParams(inputDim = self.layer1.outputDim,
#                                               poolsize = (2,2))
#             self.layerCfgs.append(self.layer1pool)
#             
#             self.layer1actf = ActivationFunctionLayerParams(inputDim = self.layer1pool.outputDim,
#                                                             activation = ReLU)
#             self.layerCfgs.append(self.layer1actf)
#             
#             l1out = self.layer1actf.outputDim
#             self.layer2 = HiddenLayerParams(inputDim = (l1out[0],l1out[1]*l1out[2]*l1out[3]),  
#                                             outputDim = (batchSize,256),  # 32
#                                             activation = ReLU)# relu
#             self.layerCfgs.append(self.layer2)
# 
#             self.layer3 = HiddenLayerParams(inputDim = self.layer2.outputDim,  
#                                             outputDim = (batchSize,nOut), # 3  
#                                             activation = None)  # linear output for the regression target
#             self.layerCfgs.append(self.layer3)
#             
#             self.outputDim = self.layer3.outputDim
#             
#             #self.layerCfgs = [self.layer0,self.layer1,self.layer2,self.layer2b,self.layer3]
#             #self.layerCfgs = [self.layer0,self.layer1,self.layer2,self.layer3]
#                              
#         if (typeID == 6):  # 
#             
#             self.layerCfgs = [] 
#             self.layer0 = ConvLayerParams(inputDim = (batchSize,nChan,wIn,hIn),  # w,h,nChannel
#                                           nFilters = 16,#10,
#                                           #filterDim = (9,9),  
#                                           filterDim = (8,8),
#                                           activation = ReLU,
#                                           poolsize = (2,2),
#                                           wInitMode = 1,  # random init
#                                           #wInitMode = 2,  # sinusoids init
#                                           wInitOrthogonal = False)
#             self.layerCfgs.append(self.layer0)
#             
#             self.layer1 = ConvLayerParams(inputDim = self.layer0.outputDim,
#                                           nFilters = 7,
#                                           filterDim = (5,5),
#                                           activation = ReLU,
#                                           poolsize = (2,2),
#                                           wInitMode = 1, # random init
#                                           #wInitMode = 2, # sinusoids init
#                                           wInitOrthogonal = False)
#             self.layerCfgs.append(self.layer1)             
#             
#             l1out = self.layer1.outputDim
#             self.layer2 = HiddenLayerParams(inputDim = (l1out[0],l1out[1]*l1out[2]*l1out[3]),  
#                                             #outputDim = (bs,500),
#                                             outputDim = (batchSize,256),  # 32
#                                             activation = ReLU)# relu
#             self.layerCfgs.append(self.layer2)
# 
#             self.layer2b = HiddenLayerParams(inputDim = self.layer2.outputDim,  
#                                              #outputDim = (bs,500),
#                                              outputDim = (batchSize,128),  # 32
#                                              activation = ReLU)# relu
#             self.layerCfgs.append(self.layer2b)
# 
#             self.layer3 = HiddenLayerParams(inputDim = self.layer2b.outputDim,  
#                                             outputDim = (batchSize,nOut), # 3  
#                                             activation = None)  # linear output for the regression target
#             self.layerCfgs.append(self.layer3)
#             
#             self.outputDim = self.layer3.outputDim
#             
#             #self.layerCfgs = [self.layer0,self.layer1,self.layer2,self.layer2b,self.layer3]
#             #self.layerCfgs = [self.layer0,self.layer1,self.layer2,self.layer3]
#                        
#                                     
#         if (typeID == 7):  # 
#             
#             self.layerCfgs = [] 
#             self.layer0 = ConvLayerParams(inputDim = (batchSize,nChan,wIn,hIn),  # w,h,nChannel
#                                           nFilters = 16,#10,
#                                           #filterDim = (9,9),  
#                                           filterDim = (8,8),
#                                           activation = ReLU,
#                                           poolsize = (2,2),
#                                           wInitMode = 1,  # random init
#                                           #wInitMode = 2,  # sinusoids init
#                                           wInitOrthogonal = False)
#             self.layerCfgs.append(self.layer0)
#             
#             self.layer1 = ConvLayerParams(inputDim = self.layer0.outputDim,
#                                           nFilters = 7,
#                                           filterDim = (5,5),
#                                           activation = ReLU,
#                                           poolsize = (2,2),
#                                           wInitMode = 1, # random init
#                                           #wInitMode = 2, # sinusoids init
#                                           wInitOrthogonal = False)
#             self.layerCfgs.append(self.layer1)             
#             
#             self.layer2 = ConvLayerParams(inputDim = self.layer1.outputDim,
#                                           nFilters = 256,
#                                           filterDim = (1,1),
#                                           activation = ReLU,
#                                           poolsize = (1,1),
#                                           poolType = -1,
#                                           wInitMode = 1, # random init
#                                           #wInitMode = 2, # sinusoids init
#                                           wInitOrthogonal = False)
#             self.layerCfgs.append(self.layer2)   
#                         
#             self.layer3 = ConvLayerParams(inputDim = self.layer2.outputDim,
#                                           nFilters = nOut,
#                                           filterDim = (1,1),
#                                           activation = None, # linear output for the regression target
#                                           poolsize = (1,1),
#                                           poolType = -1,
#                                           wInitMode = 1, # random init
#                                           #wInitMode = 2, # sinusoids init
#                                           wInitOrthogonal = False)
#             self.layerCfgs.append(self.layer3)   
#                         
#             self.outputDim = self.layer3.outputDim
            


    def readFromConfig(self):
        # TODO
        pass
    
    
    def __getstate__(self):
        
        state = dict()
        state['nwClass'] = self.nwClass
        state['batch_size'] = self.batch_size
        state['typeID'] = self.typeID
        state['layerCfgs'] = self.layerCfgs
        state['outputDim'] = self.outputDim
        return state
        
    def __setstate__(self,state):
        
        self.nwClass = state['nwClass'] 
        self.batch_size = state['batch_size'] 
        self.typeID = state['typeID'] 
        self.layerCfgs = state['layerCfgs'] 
        self.outputDim = state['outputDim'] 
        
   
    def debugPrint(self):
        
        print("Network structure {")
        for i,layerCfg in enumerate(self.layerCfgs):
            print("  Layer {} '{}': ".format(i,layerCfg.name),end="")
            layerCfg.debugPrint(4)
        print("}")

    def debugPlot(self,outfile):
        pass
        

class Network(object):
    
    def __init__(self,rng,inputVar=None,cfgParams=None):
        '''
        
        :type cfgParams: NetworkParams
        '''     
        if cfgParams is None:
            raise Exception("Cannot create a Network without config parameters (ie. cfgParams==None)")
        
        self.cfgParams = cfgParams
        self.tfComputeDescr = None 
        self.descrCompDataManager = None
        self.descrCompBatchSize = None
                
        self.setup(rng,inputVar)
        
        
    def setup(self, rng, inputVar):

        if inputVar is None:           
            inputVar = T.tensor4('x0')    # input variable
        elif isinstance(inputVar,str): 
            inputVar = T.tensor4(inputVar)    # input variable
        self.inputVar = inputVar
        
        layerInput = self.inputVar
        layers = []
        params = []
        weights = []
        #for i in xrange(len(cfgParams.layerCfgs)):
        for i,layerCfg in enumerate(self.cfgParams.layerCfgs):
            if (len(layerCfg.inputDim) < 3) and (layerInput.ndim > 2):
                layerInput = layerInput.flatten(2)
            layer = layerCfg.LayerClass(rng,
                                        inputVar=layerInput,
                                        cfgParams=layerCfg,
                                        #copyLayer=None if (twin is None) else twin.layers[i],
                                        copyLayer=None,
                                        layerNum = i)
                        
            layers.append(layer)
            params.extend(layer.params)
            weights.extend(layer.weights)
            layerInput = layer.output
            
        self.layers = layers
        self.output = layers[-1].output
        self.params = params
        self.weights = weights               
        
        

    def computeDescriptors(self,test_set,dataManager,batch_size=None):
        return self.computeDescriptorsSliding(test_set,dataManager,batch_size=batch_size) 
    
    def computeDescriptorsSliding(self,test_set,dataManager,wOut=1,hOut=1,batch_size=None):
        assert isinstance(dataManager,DataManager), "dataManager is not a DataManager but {}".format(type(dataManager))
        
        self.setupDescriptorComputation(dataManager, batch_size)
        
        floatX = theano.config.floatX  # @UndefinedVariable
        
        nSamp = test_set.numSamples
        descrLen = self.cfgParams.outputDim[1]        
        if (wOut > 1) or (hOut > 1):
            descr = numpy.zeros((nSamp,descrLen,wOut,hOut),dtype=floatX)
        else:
            descr = numpy.zeros((nSamp,descrLen),dtype=floatX)
        
        n_test_batches = nSamp / batch_size
        assert nSamp == batch_size * n_test_batches, "can only handle full mini-batches. nSamp={},n_test_batches={},batch_size={}".format(nSamp,n_test_batches,batch_size)
        
        descr_comp_start_time = time.time()
        for i in range(n_test_batches):
            idx = dataManager.makeMinibatchAvailable(test_set, i)
            descr[i*batch_size:(i+1)*batch_size] = numpy.squeeze(self.tfComputeDescr(idx))
        descr_comp_end_time = time.time()
        print ('Computing descriptors for %d samples took %.5f' % (nSamp,(descr_comp_end_time - descr_comp_start_time)) )
        
        return descr
    
    
    def setupDescriptorComputation(self,dataManager,batch_size=None):

        if batch_size == None:
            batch_size = self.cfgParams.batch_size

        if (self.tfComputeDescr is not None) and (self.descrCompDataManager == dataManager) and (self.descrCompBatchSize == batch_size):
            # no need to recompile
            return
        
        self.descrCompDataManager = dataManager
        self.descrCompBatchSize = batch_size

        self.tvIndex = T.lscalar()
        print("compiling compute_descr() ...",end="")
        if isinstance(self.inputVar,list):
            tvsData_x = dataManager.tvsData_x
            if not isinstance(tvsData_x,list):
                tvsData_x = [tvsData_x]
            
            givens_comp_descr = { iv: data[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] for (iv,data) in zip(self.inputVar,tvsData_x) }
        else:
            givens_comp_descr = { self.inputVar: dataManager.tvsData_x[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] }
        self.tfComputeDescr = theano.function(inputs = [self.tvIndex],
                                              outputs = self.output,
                                              givens = givens_comp_descr )
                             
        print("done")
        
    
    
    def save(self,folder,filePrefix):

        fileName = os.path.join(folder,filePrefix)
        
        #store layers, let them store where they saved there stuff in cfgParams
        for i,layer in enumerate(self.layers):
            layerFileName = "{}_layer{:02d}_{}".format(fileName,i,layer.cfgParams.name)
            layer.saveValues(layerFileName)
            
        #   pkl the cfgParams
        with open(fileName+".pkl",'wb') as f:
            #print(">>> type {}".format(type(self.cfgParams)))
            cPickle.dump(self.cfgParams,f,protocol=cPickle.HIGHEST_PROTOCOL)
            
            # TODO: the cfgParams need a save method, where only stuff that's really needed is stored.
            #       otherwise there is still theano related stuff in the pkl (I'm guessing it's only the activation functions)
            #     -> or probably easier: they all need __getstate__ and __setstate__, where actfunctions can be converted to string
    
    
    @classmethod
    def load(cls,folder,filePrefix,rng,inputVar=None):
        # load cfgParams
        #self.tfComputeDescr = None
        #self.descrCompDataManager = None
        #self.descrCompBatchSize = None

        fileName = os.path.join(folder,filePrefix)

        # just read the cfgParams from the pkl        
        with open(fileName+".pkl",'rb') as f:
            cfgParams = cPickle.load(f)
        
        # then recreate the network
        #self.setup(rng,inputVar)
        NetworkClass = cfgParams.nwClass
        newNetwork = NetworkClass(rng,cfgParams=cfgParams)
        return newNetwork
            
    
    @property
    def weightVals(self):
        return self.recGetWeightVals(self.params)
        
    @weightVals.setter
    def weightVals(self,value):
        self.recSetWeightVals(self.params, value)
            
    def recSetWeightVals(self,param,value):
        if isinstance(value,list):
            assert isinstance(param,list), "tried to assign a list of weights to params, which is not a list {}".format(type(param))
            assert len(param) == len(value), "tried to assign unequal list of weights {} != {}".format(len(param),len(value))
            for i in xrange(len(value)):
                self.recSetWeightVals(param[i],value[i])
        else:
            param.set_value(value)
    
    def recGetWeightVals(self,param):
        w = []
        if isinstance(param,list): 
            for p in param:
                w.append(self.recGetWeightVals(p))
        else:
            w = param.get_value()
        return w


    def getNumWeights(self):
        return self.recGetNumWeights(self.weights)
    
    def recGetNumWeights(self,weight):
        n = []
        if isinstance(weight,list): 
            for ws in weight:
                n.extend(self.recGetNumWeights(ws))
        else:
            #n = [numpy.prod(param.get_value().shape)]
            n = [weight.get_value().shape]
        return n
    
    

        