'''
Created on Jun 25, 2014

@author: wohlhart
'''
from __future__ import print_function

import theano
import theano.tensor as T
import numpy
import time
import math
import scipy.spatial.distance
import cPickle
import copy
import logging

from tnetcore.layers.logisticregression import LogisticRegression, LogisticRegressionParams
from tnetcore.data import DataManager

    
    

class NetworkTrainingParams(object):

    def __init__(self):
        self.batch_size = 128
        self.momentum = 0.9
        self.learning_rate = 0.02
        self.learning_rate_schedule_iter = 0
        self.learning_rate_schedule_factor = 0.9
        self.save_traindescr_interval = 0
        self.save_traindescr_filebase = ''
        self.weightreg_factor = 0.001  # regularization on the weights        
        self.nClasses = 2
        self.tripletThresh = 0.0
        self.tripletCostType = 0
        self.tripletPoolCostType = 0
        self.tripletPoolThresh = 0.0
        
        self.use_labels = False
        self.use_regtargets = False
        self.use_pairs = False
        self.use_triplets = False
        self.use_tripletPools = False
        
        self.debugResPath = ""
        
        
    def __str__(self):
         
        cstr = "NetworkTrainingParams("
        cstr += "batch_size={}".format(self.batch_size)
        cstr += ",momentum={}".format(self.momentum)
        cstr += ",learning_rate={}".format(self.learning_rate)
        cstr += ",weightreg_factor={}".format(self.weightreg_factor)        
        cstr += ",nClasses={}".format(self.nClasses)
        cstr += ",tripletThresh={}".format(self.tripletThresh)
        cstr += ",tripletCostType={}".format(self.tripletCostType)
        cstr += ",tripletPoolCostType={}".format(self.tripletPoolCostType)
        cstr += ",tripletPoolThresh={}".format(self.tripletPoolThresh)
 
        cstr += ",use_labels={}".format(self.use_labels)
        cstr += ",use_regtargets={}".format(self.use_regtargets)
        cstr += ",use_pairs={}".format(self.use_pairs)
        cstr += ",use_triplets={})".format(self.use_triplets)
        cstr += ",use_tripletPools={})".format(self.use_tripletPools)
         
        return cstr
        
    def debugPrint(self):
        
        cstr = "NetworkTrainingParams(\n"
        cstr += "  batch_size =       {}\n".format(self.batch_size)
        cstr += "  momentum =         {}\n".format(self.momentum)
        cstr += "  learning_rate =    {}\n".format(self.learning_rate)
        cstr += "  learning_rate_schedule_iter =    {}\n".format(self.learning_rate_schedule_iter)
        cstr += "  learning_rate_schedule_factor =    {}\n".format(self.learning_rate_schedule_factor)
        cstr += "  save_traindescr_interval =    {}\n".format(self.save_traindescr_interval)
        if self.save_traindescr_interval > 0:
            cstr += "  save_traindescr_filebase =    {}\n".format(self.save_traindescr_filebase)
        cstr += "  weightreg_factor =    {}\n".format(self.weightreg_factor)        
        cstr += "  nClasses =            {}\n".format(self.nClasses)
        cstr += "  tripletThresh =       {}\n".format(self.tripletThresh)
        cstr += "  tripletCostType =     {}\n".format(self.tripletCostType)
        cstr += "  tripletPoolCostType = {}\n".format(self.tripletPoolCostType)
        cstr += "  tripletPoolThresh =   {}\n".format(self.tripletPoolThresh)
 
        cstr += "  use_labels =      {}\n".format(self.use_labels)
        cstr += "  use_regtargets =   {}\n".format(self.use_regtargets)
        cstr += "  use_pairs =        {}\n".format(self.use_pairs)
        cstr += "  use_triplets =     {}\n".format(self.use_triplets)        
        cstr += "  use_tripletPools = {})".format(self.use_tripletPools)        
        print(cstr)


class NetworkTrainer(object):
    '''
    classdocs
    '''

    def __init__(self, descrNet, cfgParams, rng):
        '''
        Constructor
        
        :param descrnet: initialized DescriptorNet
        :param cfgParams: initialized NetworkTrainingParams 
        '''

        self.descrNet = descrNet
        self.cfgParams = cfgParams
        self.rng = rng
        
        if not isinstance(cfgParams,NetworkTrainingParams):            
            raise ValueError("cfgParams must be an instance of NetworkTrainingParams")
        
        if ((not self.cfgParams.use_labels) and 
            (not self.cfgParams.use_regtargets) and 
            (not self.cfgParams.use_pairs) and 
            (not self.cfgParams.use_triplets) and
            (not self.cfgParams.use_tripletPools)):
            raise ValueError("CONFIG ERROR: neither use_labels nor use_pairs nor use_triplets nor use_tripletPools is true.")
        
        self.tfTest = None
        self.tfTrainModel = None
        self.tfTestModelOnTrain = None
        self.tfErrAndDstDiffOnTrain = None
        self.tfEvalCost = None
        self.tfComputeDescr = None

        self.setupVariables()        
        self.setupCost()
        self.setupError()
        
    
    def setupVariables(self):
        floatX = theano.config.floatX  # @UndefinedVariable
        

        # params
        self.learning_rate = T.scalar('learning_rate',dtype=floatX) 
        self.momentum = T.scalar('momentum',dtype=floatX)
        
        # input
        self.tvIndex = T.lscalar()  # index to a [mini]batch
        #self.tvIndex.tag.test_value = 10
        self.tvX = self.descrNet.inputVar
     
        # targets
        self.tvY = T.ivector('y')
        self.tvYr = T.tensor4('yr')
        self.tvPairIdx = T.imatrix('pairIdx')
        self.tvPairLabels = T.ivector('pairLabels')
        self.tvTripletIdx = T.imatrix('tripletIdx')
        self.tvTripletThresh = T.scalar('tripletThresh')
        self.tvTripletPoolIdx = T.imatrix('tripletPoolIdx')
        self.tvTripletPoolThresh = T.scalar('tripletPoolThresh')
        self.tvPosTripletPoolSize = T.iscalar('posTripletPoolSize')
        self.tvNegTripletPoolSize = T.iscalar('negTripletPoolSize')
        
        
    def setupCost(self):
        
        dnParams = self.descrNet.cfgParams
        
        self.cost = 0
        
        if self.cfgParams.use_labels:
            # softmax layer to targets ...
            params = LogisticRegressionParams(inputDim=dnParams.outputDim,outputDim=self.cfgParams.nClasses)
            self.logregLayer = LogisticRegression(self.rng,inputVar=self.descrNet.output,cfgParams=params)
            xent = - self.y * T.log(self.logregLayer.p_y_given_x) - (1-self.tvY) * T.log(1 - self.logregLayer.p_y_given_x) # assuming the outputs are in [0..1]
            self.cost += xent.mean() # The cost to minimize = cross entropy between output and labels
            pass
        
        if self.cfgParams.use_regtargets:
            # softmax layer to targets ...
            self.sqErr = T.sqr(self.descrNet.output - self.tvYr) # .flatten(1).sum(axis=1)
            self.cost += self.sqErr.mean() 
            pass        
        
        if self.cfgParams.use_pairs:
            pair_diff_idx0 = self.tvPairIdx[:,0] 
            pair_diff_idx1 = self.tvPairIdx[:,1] 
            pair_difference = self.descrNet.output[pair_diff_idx0] - self.descrNet.output[pair_diff_idx1]
            pair_distance_sq =  T.sum(pair_difference**2,axis=1) / dnParams.outputDim[1]  # divide by num outputs, such that the max distance is 1
            
            version = 'v2'
            if version == 'v1':
                pair_p_equal = (1 - pair_distance_sq)*0.98 + 0.01  # robustify, should we do some sigmoid stuff here?
                # cost
                pair_xent = - self.tvPairLabels * T.log(pair_p_equal) - (1-self.tvPairLabels) * T.log(1-pair_p_equal) # Cross-entropy loss function
                self.cost += pair_xent.mean() # The cost to minimize = cross entropy

            if version == 'v2':
                self.pair_neg_margin = 0.6
                eps = 1e-6
                pair_cost = self.tvPairLabels*pair_distance_sq + (1-self.tvPairLabels)*T.sqr(T.maximum(0,self.pair_neg_margin - T.sqrt(pair_distance_sq+eps)))                
                self.cost += pair_cost.mean() 
                
                # TODO: how do we compute p_equal in this case?
                distance = T.sqrt(pair_distance_sq)
                pair_p_equal = (1 - distance)*0.98 + 0.01              
                pair_xent = - self.tvPairLabels * T.log(pair_p_equal) - (1-self.tvPairLabels) * T.log(1-pair_p_equal) # Cross-entropy loss function
                
#             if version == 'v3':
#                 self.pair_neg_margin = 0.6
#                 per_pair_cost = pairLabels*pair_distance_sq + (1-pairLabels)*(1 - pair_distance_sq)
#                 self.cost = per_pair_cost.mean()

            # store
            self.pair_xent = pair_xent                
            self.pair_p_equal = pair_p_equal 
            self.pair_difference = pair_difference
            self.pair_distance_sq = pair_distance_sq
            self.pair_diff_idx0 = pair_diff_idx0
            self.pair_diff_idx1 = pair_diff_idx1 
            
            
        if self.cfgParams.use_triplets:    
            tripletThresh = self.tvTripletThresh
            
            triplet_idx0 = self.tvTripletIdx[:,0] 
            triplet_idx1 = self.tvTripletIdx[:,1] 
            triplet_idx2 = self.tvTripletIdx[:,2] 
            difference_sim = self.descrNet.output[triplet_idx0] - self.descrNet.output[triplet_idx1]
            difference_dis = self.descrNet.output[triplet_idx0] - self.descrNet.output[triplet_idx2]
            eps = 1e-6
            if self.cfgParams.tripletCostType == 0:
                distance_sim_sq =  T.sum(difference_sim**2,axis=1) / dnParams.outputDim[1]  # divide by num outputs, such that the max distance is 1 (if each output is restricted to 0..1)
                distance_dis_sq =  T.sum(difference_dis**2,axis=1) / dnParams.outputDim[1]
                dstdiff = (distance_sim_sq - distance_dis_sq) / (distance_sim_sq + distance_dis_sq + eps) 
            elif self.cfgParams.tripletCostType == 1:
                distance_sim_sq =  T.sum(difference_sim**2,axis=1) / dnParams.outputDim[1]  # divide by num outputs, such that the max distance is 1
                distance_dis_sq =  T.sum(difference_dis**2,axis=1) / dnParams.outputDim[1]
                dstdiff = (distance_sim_sq - distance_dis_sq + tripletThresh) / (distance_sim_sq + distance_dis_sq + eps) 
            elif self.cfgParams.tripletCostType == 2:
                distance_sim_sq =  T.sum(difference_sim**2,axis=1) / dnParams.outputDim[1]  # divide by num outputs, to make it easier to balance with weight regularization
                distance_dis_sq =  T.sum(difference_dis**2,axis=1) / dnParams.outputDim[1]
                distance_sim =  T.sqrt(distance_sim_sq)
                distance_dis =  T.sqrt(distance_dis_sq)
                #dstdiff = (distance_sim*tripletThresh - distance_dis) / (distance_sim) 
                #dstdiff = T.switch(T.gt(distance_sim,0), tripletThresh - distance_dis / distance_sim, 0)
                #dstdiff = tripletThresh - distance_dis / T.maximum(distance_sim,numpy.array(eps,dtype=numpy.float32))
                dstdiff = (distance_sim_sq - distance_dis_sq + tripletThresh) / (distance_sim_sq + distance_dis_sq + eps) 
            elif self.cfgParams.tripletCostType == 3:
                distance_sim_sq =  T.sum(difference_sim**2,axis=1) / dnParams.outputDim[1]  # divide by num outputs, to make it easier to balance with weight regularization
                distance_dis_sq =  T.sum(difference_dis**2,axis=1) / dnParams.outputDim[1]
                dstdiff = (distance_sim_sq + tripletThresh - distance_dis_sq) / T.maximum(distance_sim_sq,numpy.array(eps,dtype=numpy.float32))
            elif self.cfgParams.tripletCostType == 4:
                distance_sim_sq =  T.sum(difference_sim**2,axis=1) / dnParams.outputDim[1]  # divide by num outputs, to make it easier to balance with weight regularization
                distance_dis_sq =  T.sum(difference_dis**2,axis=1) / dnParams.outputDim[1]
                dstdiff = (distance_sim_sq + tripletThresh - distance_dis_sq) / T.maximum(distance_sim_sq,tripletThresh)
            elif self.cfgParams.tripletCostType == 5:    
                #  try this one, looks nice in plot: (numpy.maximum((mx+eps),m)+ m - my+eps)/numpy.maximum(mx+eps,m)
                eps = tripletThresh/10.
                distance_sim_sq =  T.sum(difference_sim**2,axis=1) / dnParams.outputDim[1] +eps*eps  # divide by num outputs, to make it easier to balance with weight regularization
                distance_dis_sq =  T.sum(difference_dis**2,axis=1) / dnParams.outputDim[1] +eps*eps  # +eps because  the grad of a sqrt(sqdst) is NaN if sqdst = 0 
                distance_sim =  T.sqrt(distance_sim_sq)  
                distance_dis =  T.sqrt(distance_dis_sq)
                dstdiff = (T.maximum(distance_sim,tripletThresh) + tripletThresh - distance_dis) / T.maximum(distance_sim,tripletThresh)
            elif self.cfgParams.tripletCostType == 6:    
                #  try this one, looks nice in plot: (numpy.maximum((mx+eps),m)+ m - my+eps)/numpy.maximum(mx+eps,m)
                eps = tripletThresh/100.
                distance_sim =  T.sqrt(T.sum((abs(difference_sim)+eps)**2,axis=1) / dnParams.outputDim[1])  # divide by num outputs, to make it easier to balance with weight regularization
                distance_dis =  T.sqrt(T.sum((abs(difference_dis)+eps)**2,axis=1) / dnParams.outputDim[1]) # +eps because  the grad of a sqrt(sqdst) is NaN if sqdst = 0 
                #dstdiff = (T.maximum(distance_sim,tripletThresh) + tripletThresh - distance_dis) / T.maximum(distance_sim,tripletThresh)
                dstdiff = 1.0 - distance_dis/(distance_sim + tripletThresh)                
            elif self.cfgParams.tripletCostType == 7:    
                #  cost from Wang et al. "Learning fine-grained image similarity with deep ranking.", CVPR 2014
                eps = tripletThresh/100.
                distance_sim =  T.sqrt(T.sum((abs(difference_sim)+eps)**2,axis=1) / dnParams.outputDim[1])  # divide by num outputs, to make it easier to balance with weight regularization
                distance_dis =  T.sqrt(T.sum((abs(difference_dis)+eps)**2,axis=1) / dnParams.outputDim[1]) # +eps because  the grad of a sqrt(sqdst) is NaN if sqdst = 0 
                dstdiff = tripletThresh + distance_sim - distance_dis
            elif self.cfgParams.tripletCostType == 8:    
                #  another one, like the one used for CVPR, but where the cost also rises linearly for increasing similar pair distance 
                eps = tripletThresh/100.
                distance_sim =  T.sqrt(T.sum((abs(difference_sim)+eps)**2,axis=1) / dnParams.outputDim[1])  # divide by num outputs, to make it easier to balance with weight regularization
                distance_dis =  T.sqrt(T.sum((abs(difference_dis)+eps)**2,axis=1) / dnParams.outputDim[1]) # +eps because  the grad of a sqrt(sqdst) is NaN if sqdst = 0 
                dstdiff = (1+distance_sim)*(1.0 - distance_dis/(distance_sim + tripletThresh))                
                
            triplet_cost = dstdiff * (dstdiff > 0.0) #T.maximum(0,)  # if dis>sim, sim-dis is negative -> okay, otherwise penalty
            self.cost += T.mean(triplet_cost)
            
            self.triplet_cost = triplet_cost
            self.triplet_dst_diff = dstdiff       
            self.triplet_dst_diff_raw = T.sqrt(T.sum(difference_sim**2,axis=1)) - T.sqrt(T.sum(difference_dis**2,axis=1))

        if self.cfgParams.use_tripletPools:   
            #npos = self.posTripletPoolSize
            #nneg = self.negTripletPoolSize             
            npos = self.tvPosTripletPoolSize
            nneg = self.tvNegTripletPoolSize             
            triplet_idx0 = self.tvTripletPoolIdx[:,0] 
            triplet_idx1 = self.tvTripletPoolIdx[:,1:1+npos]
            triplet_idx2 = self.tvTripletPoolIdx[:,1+npos:1+npos+nneg]
            tripletPoolThresh = self.tvTripletPoolThresh
            
            #   calc a dist for every and then the min over the first pool (similars)
            anchor_descr = self.descrNet.output[triplet_idx0]
            #theano.printing.debugprint(anchor_descr,print_type=True)
            pos_descrs = self.descrNet.output[triplet_idx1]
            neg_descrs = self.descrNet.output[triplet_idx2]
            difference_sim = anchor_descr[:,numpy.newaxis,:] - pos_descrs 
            difference_dis = anchor_descr[:,numpy.newaxis,:] - neg_descrs
            eps = 1e-6
            if self.cfgParams.tripletPoolCostType == 6:    
                #  try this one, looks nice in plot: (numpy.maximum((mx+eps),m)+ m - my+eps)/numpy.maximum(mx+eps,m)
                eps = tripletPoolThresh/100.
                min_distance_sim =  T.sqrt(T.min( T.sum((abs(difference_sim)+eps)**2,axis=1) , axis=0) / dnParams.outputDim[1])  # divide by num outputs, to make it easier to balance with weight regularization
                distance_dis =  T.sqrt(T.sum((abs(difference_dis)+eps)**2,axis=1) / dnParams.outputDim[1]) # +eps because  the grad of a sqrt(sqdst) is NaN if sqdst = 0 
                #dstdiff = (T.maximum(distance_sim,tripletThresh) + tripletThresh - distance_dis) / T.maximum(distance_sim,tripletThresh)
                dstdiff = 1.0 - distance_dis/(min_distance_sim + tripletPoolThresh)    
                raise ValueError("not yet implemented")
                             
            elif self.cfgParams.tripletPoolCostType == 7:    
                #  cost from Wang et al. "Learning fine-grained image similarity with deep ranking.", CVPR 2014
                eps = tripletPoolThresh/100.
                distance_sim = T.sqrt( T.sum((abs(difference_sim)+eps)**2,axis=2)  / dnParams.outputDim[1])  # divide by num outputs, to make it easier to balance with weight regularization
                min_distance_sim =  T.min(distance_sim, axis=1)  
                distance_dis =  T.sqrt(T.sum((abs(difference_dis)+eps)**2,axis=2) / dnParams.outputDim[1]) # +eps because  the grad of a sqrt(sqdst) is NaN if sqdst = 0
                dstdiff = tripletPoolThresh + min_distance_sim[:,numpy.newaxis] - distance_dis 
                
            triplet_pool_cost = dstdiff * (dstdiff > 0.0) #T.maximum(0,)  # if dis>sim, sim-dis is negative -> okay, otherwise penalty
            self.cost += T.mean(triplet_pool_cost)
            
            # store intermediate steps for debug functions
            self.triplet_poool_cost = triplet_pool_cost
            self.tripletPool_dst_diff = dstdiff       
            #self.tripletPool_dst_diff_raw = dstdiff
            #self.triplet_dst_diff_raw = T.sqrt(T.sum(difference_sim**2,axis=1)) - T.sqrt(T.sum(difference_dis**2,axis=1))
                 
            
        # weight vector length for regularization (weight decay)       
        totalWeightVectorLength = 0
        for W in self.descrNet.weights:
            #totalWeightVectorLength += (W**2).sum()
            #totalWeightVectorLength += (W**2).sum() / T.prod(W.shape)
            totalWeightVectorLength += (W**2).mean() 
        self.cost += self.cfgParams.weightreg_factor * totalWeightVectorLength # + weight vector norm

        # create a list of gradients for all model parameters
        self.params = self.descrNet.params
        self.grads = T.grad(self.cost, self.params)
        
    def setupError(self):
        # predictions and errors
        self.errors = 0
        self.nsamp_tested = 0
        if self.cfgParams.use_labels:
            self.errors += self.logregLayer.errors
            self.nsamp_tested += self.tvY.shape[0]

        if self.cfgParams.use_regtargets:
            self.errors += T.sum(self.sqErr > 0) # this is stupid. there is no real discrete error count for regression
            
        if self.cfgParams.use_pairs:
            y_pred_pairs = T.switch(T.gt(self.pair_p_equal,0.5), 1, 0)
            pairs_errors = T.neq(y_pred_pairs, self.tvPairLabels)
            self.errors += pairs_errors.sum()
            numPairs = self.tvPairIdx.shape[0]        
            self.nsamp_tested += numPairs
            
        if self.cfgParams.use_triplets:
            self.y_pred_triple = self.triplet_dst_diff <= 0
            #self.errors += 1 - y_pred_triple
            self.errors += (self.triplet_dst_diff > 0).sum()
            numTriplets = self.tvTripletIdx.shape[0]
            self.nsamp_tested += numTriplets

        if self.cfgParams.use_tripletPools:
            self.y_pred_triple = self.tripletPool_dst_diff <= 0
            #self.errors += 1 - y_pred_triple
            self.errors += (self.tripletPool_dst_diff > 0).sum()
            numTripletPools = self.tvTripletPoolIdx.shape[0]
            numNegTripletsPerPool = self.tvTripletPoolIdx.shape[1] // 2
            self.nsamp_tested += numTripletPools*numNegTripletsPerPool

                    
       
    def setDataAndCompileFunctions(self,train_data,val_data,dataManager,compileDebugFcts=False):
        assert isinstance(dataManager,DataManager), "datamanager is no DataManager but '{}'".format(dataManager)
        if ((self.cfgParams.use_labels != dataManager.cfg.use_labels) or 
            (self.cfgParams.use_regtargets != dataManager.cfg.use_regtargets) or 
            (self.cfgParams.use_pairs != dataManager.cfg.use_pairs) or 
            (self.cfgParams.use_triplets != dataManager.cfg.use_triplets) or
            (self.cfgParams.use_tripletPools != dataManager.cfg.use_tripletPools)):
            raise ValueError("dataManager is configured differently than trainer")

        self.setData(train_data,val_data,dataManager)
        self.compileFunctions(compileDebugFcts)
        
        
    def setData(self,train_data,val_data,dataManager):        
        '''
        @type dataManager: DataManager
        '''            
        batch_size = self.cfgParams.batch_size
        
        self.traindataDB = train_data
        self.valdataDB = val_data  

        #--------------------------------------------
        self.dataManager = dataManager
        dataManager.makeMinibatchAvailable(train_data, 0)
                
        
        # store info
        self.n_train_batches = train_data.numSamples / self.cfgParams.batch_size
        self.n_val_batches = val_data.numSamples / self.cfgParams.batch_size
            
    
    
    def compileFunctions(self,compileDebugFcts=False):
        
        #debug
        print("n_training_batches: {} ".format(self.n_train_batches))
        if self.cfgParams.use_pairs:
            print("train_data_pairIdx shape {}".format(self.traindataDB.pairIdx.shape))
            print("train_data_pairLabels shape {}".format(self.traindataDB.pairLabels.shape))
        if self.cfgParams.use_triplets:
            print("train_data_tripletIdx shape {}".format(self.traindataDB.tripletIdx.shape))
        if self.cfgParams.use_tripletPools:
            print("train_data_tripletPoolIdx shape {}".format(self.traindataDB.tripletPoolIdx.shape))
            
        # TRAIN
        self.setupTrain()
        
        self.setupErrAndDstDiffOnTrain()
        
        self.setupComputeDescr()
        
        #
        if compileDebugFcts:
            self.setupDebugFunctions()
         
        # VALIDATE
        self.setupEvalCosts()
               
        self.setupTest()
        
        
    def setupTest(self):
        print("compiling test() ... ",end="")
        test_inputs = copy.copy(self.tvX) if isinstance(self.tvX,list) else [self.tvX]
        if self.cfgParams.use_labels:
            test_inputs.append(self.tvY)
        if self.cfgParams.use_regtargets:
            test_inputs.append(self.tvYr)
        if self.cfgParams.use_pairs:
            test_inputs.append(self.tvPairIdx)
            test_inputs.append(self.tvPairLabels)
        if self.cfgParams.use_triplets:
            test_inputs.append(self.tvTripletIdx)
            test_inputs.append(self.tvTripletThresh)  #? needed
        if self.cfgParams.use_tripletPools:
            test_inputs.append(self.tvTripletPoolIdx)            
            test_inputs.append(self.tvTripletPoolThresh)
            test_inputs.append(self.tvPosTripletPoolSize)  
            test_inputs.append(self.tvNegTripletPoolSize)    
        
        self.tfTest = theano.function(inputs = test_inputs,
                                      outputs = self.errors)
        print("done.")
        

        
    def setupTrain(self):        
        # train_model is a function that updates the model parameters by SGD
        
        # from: https://github.com/gwtaylor/theano-rnn/blob/master/rnn.py
        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction        
        floatX = theano.config.floatX  # @UndefinedVariable
        
        self.last_param_update = {}
        for param in self.params:
            initVals = numpy.zeros(param.get_value(borrow=True).shape, dtype=floatX)
            self.last_param_update[param] = theano.shared(initVals)        

        # Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i],grads[i]) pairs.
        updates = []
        for param_i, grad_i in zip(self.params, self.grads):
            last_upd = self.last_param_update[param_i]
            upd = self.momentum * last_upd - self.learning_rate * grad_i
            updates.append((param_i, param_i + upd))  
            updates.append((last_upd, upd))
            #updates.append((param_i, param_i - self.cfgParams.learning_rate * grad_i))        

        dm = self.dataManager
        
        batch_size = self.cfgParams.batch_size
        if isinstance(self.tvX,list):
            givens_train = { tv: data[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] for (tv,data) in zip(self.tvX,dm.tvsData_x) }
        else:
            givens_train = { self.tvX: dm.tvsData_x[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] }
        if self.cfgParams.use_labels:
            givens_train[self.tvY] = dm.tvsData_y[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size]
        if self.cfgParams.use_regtargets:
            givens_train[self.tvYr] = dm.tvsData_yr[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size]

        if self.cfgParams.use_pairs:
            batch_idx_size = dm.nPairsPerBatch
            print("batch_idx_size {}".format(batch_idx_size))
            givens_train[self.tvPairIdx] = dm.tvsData_pairIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size] 
            givens_train[self.tvPairLabels] = dm.tvsData_pairLabels[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size]
            
        if self.cfgParams.use_triplets:
            batch_idx_size = dm.nTripletsPerBatch
            givens_train[self.tvTripletIdx] = dm.tvsData_tripletIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size]
            givens_train[self.tvTripletThresh] = numpy.array(self.cfgParams.tripletThresh).astype(floatX) 
            print("triplet info: {}".format(dm.nTripletsPerBatch))

        if self.cfgParams.use_tripletPools:
            batch_idx_size = dm.nTripletPoolsPerBatch
            givens_train[self.tvTripletPoolIdx] = dm.tvsData_tripletPoolIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size]
            givens_train[self.tvTripletPoolThresh] = numpy.array(self.cfgParams.tripletPoolThresh).astype(floatX) 
            givens_train[self.tvPosTripletPoolSize] = numpy.array(dm.posTripletPoolSizeVal).astype(numpy.int32)
            givens_train[self.tvNegTripletPoolSize] = numpy.array(dm.negTripletPoolSizeVal).astype(numpy.int32)
        
        #givens_test_on_train = givens_train
        givens_train[self.momentum] = numpy.array(self.cfgParams.momentum).astype(floatX) 
        
        print("compiling train_model() ... ",end="")
        self.tfTrainModel = theano.function(inputs = [self.tvIndex,self.learning_rate],
                                            outputs = self.cost,
                                            updates = updates,
                                            givens = givens_train) #mode='DebugMode')
        print("done.")
        
        
        print("compiling test_model_on_train() ... ",end="")
        batch_size = self.cfgParams.batch_size
        if isinstance(self.tvX,list):
            givens_test_on_train = { tv: data[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] for (tv,data) in zip(self.tvX,dm.tvsData_x) }
        else:
            givens_test_on_train = { self.tvX: dm.tvsData_x[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] }
        if self.cfgParams.use_labels:
            givens_test_on_train[self.tvY] = dm.tvsData_y[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size]
        if self.cfgParams.use_regtargets:
            givens_test_on_train[self.tvYr] = dm.tvsData_yr[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size]
             
        if self.cfgParams.use_pairs:
            batch_idx_size = dm.nPairsPerBatch
            givens_test_on_train[self.tvPairIdx] = dm.tvsData_pairIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size] 
            givens_test_on_train[self.tvPairLabels] = dm.tvsData_pairLabels[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size]
            
        if self.cfgParams.use_triplets:
            batch_idx_size = dm.nTripletsPerBatch
            givens_test_on_train[self.tvTripletIdx] = dm.tvsData_tripletIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size]
            givens_test_on_train[self.tvTripletThresh] = numpy.array(self.cfgParams.tripletThresh).astype(floatX)

        if self.cfgParams.use_tripletPools:
            batch_idx_size = dm.nTripletPoolsPerBatch
            givens_test_on_train[self.tvTripletPoolIdx] = dm.tvsData_tripletPoolIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size]
            givens_test_on_train[self.tvTripletPoolThresh] = numpy.array(self.cfgParams.tripletPoolThresh).astype(floatX)
            givens_test_on_train[self.tvPosTripletPoolSize] = numpy.array(dm.posTripletPoolSizeVal).astype(numpy.int32)
            givens_test_on_train[self.tvNegTripletPoolSize] = numpy.array(dm.negTripletPoolSizeVal).astype(numpy.int32)
        
        percWrong = self.errors/T.cast(self.nsamp_tested,'float32')
        
        self.tfTestModelOnTrain = theano.function(inputs = [self.tvIndex],
                                                  outputs = percWrong,
                                                  givens = givens_test_on_train )
        print("done.")


    def setupErrAndDstDiffOnTrain(self):
        
        floatX = theano.config.floatX  # @UndefinedVariable
        dm = self.dataManager
        
        print("compiling tfErrAndDstDiffOnTrain() ... ",end="")
        batch_size = self.cfgParams.batch_size
        if isinstance(self.tvX,list):
            givens_test_on_train = { tv: data[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] for (tv,data) in zip(self.tvX,dm.tvsData_x) }
        else:
            givens_test_on_train = { self.tvX: dm.tvsData_x[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] }
        if self.cfgParams.use_labels:
            givens_test_on_train[self.tvY] = dm.tvsData_y[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size]
        if self.cfgParams.use_regtargets:
            givens_test_on_train[self.tvYr] = dm.tvsData_yr[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size]
             
        if self.cfgParams.use_pairs:
            batch_idx_size = dm.nPairsPerBatch
            givens_test_on_train[self.tvPairIdx] = dm.tvsData_pairIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size] 
            givens_test_on_train[self.tvPairLabels] = dm.tvsData_pairLabels[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size]
            
        if self.cfgParams.use_triplets:
            batch_idx_size = dm.nTripletsPerBatch
            givens_test_on_train[self.tvTripletIdx] = dm.tvsData_tripletIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size]
            givens_test_on_train[self.tvTripletThresh] = numpy.array(self.cfgParams.tripletThresh).astype(floatX)
                
        if self.cfgParams.use_tripletPools:
            batch_idx_size = dm.nTripletPoolsPerBatch
            givens_test_on_train[self.tvTripletPoolIdx] = dm.tvsData_tripletPoolIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size]
            givens_test_on_train[self.tvTripletPoolThresh] = numpy.array(self.cfgParams.tripletPoolThresh).astype(floatX)
            givens_test_on_train[self.tvPosTripletPoolSize] = numpy.array(dm.posTripletPoolSizeVal).astype(numpy.int32)
            givens_test_on_train[self.tvNegTripletPoolSize] = numpy.array(dm.negTripletPoolSizeVal).astype(numpy.int32)

        outputs = [self.errors]
        if self.cfgParams.use_triplets:
            outputs.append(self.triplet_dst_diff_raw)
        if self.cfgParams.use_tripletPools:
            outputs.append(self.tripletPool_dst_diff)
               
        self.tfErrAndDstDiffOnTrain = theano.function(inputs = [self.tvIndex],
                                                      outputs = outputs,
                                                      givens = givens_test_on_train )
        print("done.")


    def setupEvalCosts(self):
         
        floatX = theano.config.floatX  # @UndefinedVariable
        dm = self.dataManager
        
        batch_size = self.cfgParams.batch_size
        if isinstance(self.tvX,list):
            givens_evalcost = { tv: data[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] for (tv,data) in zip(self.tvX,dm.tvsData_x) }
        else:
            givens_evalcost = { self.tvX: dm.tvsData_x[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] }
        if self.cfgParams.use_labels:
            givens_evalcost[self.tvY] = dm.tvsData_y[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size]
        if self.cfgParams.use_regtargets:
            givens_evalcost[self.tvYr] = dm.tvsData_yr[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size]
              
        if self.cfgParams.use_pairs:
            batch_idx_size = dm.nPairsPerBatch
            givens_evalcost[self.tvPairIdx] = dm.tvsData_pairIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size] 
            givens_evalcost[self.tvPairLabels] = dm.tvsData_pairLabels[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size]
             
        if self.cfgParams.use_triplets:
            batch_idx_size = dm.nTripletsPerBatch
            givens_evalcost[self.tvTripletIdx] = dm.tvsData_tripletIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size]
            givens_evalcost[self.tvTripletThresh] = numpy.array(self.cfgParams.tripletThresh).astype(floatX)
             
        if self.cfgParams.use_tripletPools:
            batch_idx_size = dm.nTripletPoolsPerBatch
            givens_evalcost[self.tvTripletPoolIdx] = dm.tvsData_tripletPoolIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size]
            givens_evalcost[self.tvTripletPoolThresh] = numpy.array(self.cfgParams.tripletPoolThresh).astype(floatX)
            givens_evalcost[self.tvPosTripletPoolSize] = numpy.array(dm.posTripletPoolSizeVal).astype(numpy.int32)
            givens_evalcost[self.tvNegTripletPoolSize] = numpy.array(dm.negTripletPoolSizeVal).astype(numpy.int32)
            
        print("compiling eval_cost() ... ",end="")
        self.tfEvalCost = theano.function(inputs = [self.tvIndex],
                                          outputs = [self.cost,self.errors],
                                          givens = givens_evalcost )
        print("done.")

        
            
    def setupComputeDescr(self):
        batch_size = self.cfgParams.batch_size
        dm = self.dataManager
        print("compiling compute_descr() ... ",end="")
        if isinstance(self.tvX,list):
            givens_comp_descr = { tv: data[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] for (tv,data) in zip(self.tvX,dm.tvsData_x) }
        else:
            givens_comp_descr = { self.tvX: dm.tvsData_x[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] }
        self.tfComputeDescr = theano.function(inputs = [self.tvIndex],
                                              outputs = self.descrNet.output,
                                              givens = givens_comp_descr )
        print("done.")
        
       
                    
        
    def train(self,n_epochs=50,iterOffset=0):
        
        #wvals = []
            
        # training procedure:
        #   n epochs, batch_size, validation
        batch_size = self.cfgParams.batch_size
        descrLen = self.descrNet.cfgParams.outputDim[1]
        
        n_train_batches = self.n_train_batches
        n_val_batches = self.n_val_batches
        #n_test_batches = self.n_test_batches
        
        # early-stopping parameters
        #patience = 128*10  # look as this many examples regardless
        #patience = n_epochs*n_train_batches/4  # look as this many batches regardless (do at least half of the epochs we were asked for)
        patience = n_epochs*n_train_batches  # forget about the patience thing. just run to the end. weights are not taken anyway if not better on validation set
        patience_increase = 2  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                        # considered significant

        #validation_frequency = min(n_train_batches, patience / 2)
                                        # go through this many minibatches before checking the network on the validation set; 
                                        # in this case we check every epoch
        validation_frequency = min(n_train_batches*10, patience / 2)
                                        # go through this many minibatches before checking the network on the validation set; 
                                        # in this case we check after 10 epochs
        print("validation_frequency {}".format(validation_frequency))

        learning_rate = self.cfgParams.learning_rate
        learning_rate_schedule_iter = self.cfgParams.learning_rate_schedule_iter
        learning_rate_schedule_factor = self.cfgParams.learning_rate_schedule_factor
        save_traindescr_interval = self.cfgParams.save_traindescr_interval
        save_traindescr_filebase = self.cfgParams.save_traindescr_filebase
        
        #best_params = None
        best_validation_loss = numpy.inf
        #lastPatienceCheck_best_validation_loss = -1
        bestParams = None
                 
        start_time = time.clock()
                                    
        train_costs = []
        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):        
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                # iteration number ("we just did the iter_count-th batch")
                iter_count = (epoch - 1) * n_train_batches + minibatch_index
                
                #print("batch {}".format(minibatch_index))
                
                #### DO TRAIN
#                 if self.macroBatchSize > 0:
#                     macroBatchIdx = numpy.floor(minibatch_index / self.macroBatchSize).astype(numpy.int)
#                     self.setMacroBatchData(self.traindataDB,macroBatchIdx)
#                     miniBatchIdx = numpy.mod(minibatch_index,self.macroBatchSize)
#                 else:
#                     miniBatchIdx = minibatch_index
#                     macroBatchIdx = 0
                miniBatchIdx = self.dataManager.makeMinibatchAvailable(self.traindataDB,minibatch_index)
                    
                #print("train minibatch {}, in macrobatch {}".format(miniBatchIdx,numpy.floor(minibatch_index / self.macroBatchSize).astype(numpy.int)))
                minibatch_avg_cost = self.tfTrainModel(miniBatchIdx,learning_rate)
                
                if math.isnan(minibatch_avg_cost):
                    print("minibatch {0:4d}, average cost: NaN".format(minibatch_index))
                    # check which vars are nan
                    self.checkNaNs()
                    
                    assert(False)
                    
                #print("iter {:4d} ({:4d}), epoch {:4d}, minibatch {:4d}, macrobatch {:4d}, average cost: {}".format(iter_count,iterOffset+iter_count,epoch,minibatch_index,macroBatchIdx,minibatch_avg_cost))
                print("iter {:4d} ({:4d}), epoch {:4d}, minibatch {:4d}, average cost: {}".format(iter_count,iterOffset+iter_count,epoch,minibatch_index,minibatch_avg_cost))
                
                train_costs.append(minibatch_avg_cost)
                
                # save descriptors
                if (save_traindescr_interval > 0) and ((numpy.mod(iter_count,save_traindescr_interval) == 0) or (epoch == 1)):
                    descr = numpy.zeros((n_train_batches*batch_size,descrLen))
                    for i in xrange(n_train_batches):
#                         if self.macroBatchSize > 0:
#                             self.setMacroBatchData(self.traindataDB,numpy.floor(i / self.macroBatchSize).astype(numpy.int))
#                             miniBatchIdx = numpy.mod(i,self.macroBatchSize)
#                         else:
#                             miniBatchIdx = i
                        miniBatchIdx = self.dataManager.makeMinibatchAvailable(self.traindataDB,i)
                        descr[i*batch_size:(i+1)*batch_size] = self.tfComputeDescr(miniBatchIdx)                        
                      
                    with open(save_traindescr_filebase.format(iterOffset + iter_count),'wb') as f:
                        cPickle.dump({'descr':descr,'iter_count':iter_count,'iterOffset':iterOffset},f,protocol=cPickle.HIGHEST_PROTOCOL)
                      
                    # 
                
                # learning rate schedule
                if (iter_count > 0) and (numpy.mod(iter_count,learning_rate_schedule_iter) == 0):
                    learning_rate *= learning_rate_schedule_factor
                    print("reduced learning_rate: {}".format(learning_rate))
    
                # validate
                if ((iter_count + 1) % validation_frequency == 0) or (patience <= iter_count):
                    #
#                     if storeFilters:
#                         wval = self.descrNet.layers[0].W.get_value()
#                         wvals.append(wval)
                    
                    # DEBUG save validation set descriptor visualization                        
                    #valdescrs = self.computeValDescriptors()
                    #vis3DDescriptors2(valdescrs,self.valdataDB.y,"{}{}_valDescr_{}.png".format(self.cfgParams.debugResPath,nowStr(),iter_count))

                    # compute errors on training data                    
                    this_train_errors = numpy.zeros(n_train_batches)
                    for i in xrange(n_train_batches):
#                         if self.macroBatchSize > 0:
#                             self.setMacroBatchData(self.traindataDB,numpy.floor(i / self.macroBatchSize).astype(numpy.int))
#                             miniBatchIdx = numpy.mod(i,self.macroBatchSize)
#                         else:
#                             miniBatchIdx = i
                        miniBatchIdx = self.dataManager.makeMinibatchAvailable(self.traindataDB,i)     
                        #print("test minibatch {}, in macrobatch {}".format(miniBatchIdx,numpy.floor(i / self.macroBatchSize).astype(numpy.int)))
                        bTrainPercWrong = self.tfTestModelOnTrain(miniBatchIdx)                        
                        this_train_errors[i] = bTrainPercWrong
                    this_train_errors = numpy.mean(this_train_errors)
                    
                    print('epoch %i, minibatch %i/%i, train error %f %%' % \
                        (epoch, minibatch_index + 1, n_train_batches, this_train_errors * 100.))
                    
#                     # compute zero-one loss on validation set
#                     validation_losses = [self.validation_errors(i) for i in xrange(n_val_batches)]
#                     this_validation_loss = numpy.mean(validation_losses)
                    # compute cost on validation set
                    validation_losses = numpy.zeros(n_val_batches)
                    for i in xrange(n_val_batches):
#                         if self.macroBatchSize > 0:
#                             self.setMacroBatchData(self.valdataDB,numpy.floor(i / self.macroBatchSize).astype(numpy.int))
#                             miniBatchIdx = numpy.mod(i,self.macroBatchSize)
#                         else:
#                             miniBatchIdx = i   
                        miniBatchIdx = self.dataManager.makeMinibatchAvailable(self.valdataDB,i)                     
                        bValCost,bValErrors = self.tfEvalCost(miniBatchIdx)
                        validation_losses[i] = bValCost
                    this_validation_loss = numpy.mean(validation_losses)
    
                    print('epoch %i, minibatch %i/%i, validation cost %f ' % \
                        (epoch, minibatch_index + 1, n_train_batches, this_validation_loss))
    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter_count * patience_increase)
    
                        best_validation_loss = this_validation_loss
                        
                        # TODO store best parameters
                        print("best validation loss so far -> store network weights")
                        bestParams = self.descrNet.weightVals

                if patience <= iter_count:
                    done_looping = True
                    break

        end_time = time.clock()
        print(('Optimization complete with best validation score of %f,') % (best_validation_loss))
        print ('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)) )
        
        if bestParams is not None:
            self.descrNet.weightVals = bestParams
        
        progressData = dict() 
        progressData['train_costs'] = train_costs
        return (progressData,iter_count)
    
    
    def checkNaNs(self):
        
        #floatX = theano.config.floatX  # @UndefinedVariable
        
        for param_i in self.params: 
            if numpy.any(numpy.isnan(param_i.get_value())):
                print("NaN in weights")
                
        for lpu in self.last_param_update:
            if numpy.any(numpy.isnan(lpu.get_value())):
                print("NaN in last_param_update")
                
        n_train_batches = self.n_train_batches
        #batch_size = self.cfgParams.batch_size
        for i in range(n_train_batches):
            miniBatchIdx = self.dataManager.makeMinibatchAvailable(self.traindataDB,i)
            descr = self.tfComputeDescr(miniBatchIdx)
            
            if numpy.any(numpy.isnan(descr)):
                print("NaN in descriptor in batch {}".format(i))
            else:
                print("No NaNs in descriptors in batch {}".format(i))
        
            
    def computeDescriptors(self,test_set):
        
        batch_size = self.cfgParams.batch_size
        nSamp = test_set.numSamples
        
        descrLen = self.descrNet.cfgParams.outputDim[1]
        descr = numpy.zeros((nSamp,descrLen))
        
        n_test_batches = nSamp / batch_size
        
        assert nSamp == batch_size * n_test_batches, "can only handle full mini-batches. nSamp={},n_test_batches={},batch_size={}".format(nSamp,n_test_batches,batch_size)
        for i in range(n_test_batches):
#             if self.macroBatchSize > 0:
#                 self.setMacroBatchData(test_set,numpy.floor(i / self.macroBatchSize).astype(numpy.int))
#                 miniBatchIdx = numpy.mod(i,self.macroBatchSize)
#             else:
#                 miniBatchIdx = i
            miniBatchIdx = self.dataManager.makeMinibatchAvailable(test_set,i)
            #print("compute descriptors for minibatch {} ({})".format(i,miniBatchIdx))
            descr[i*batch_size:(i+1)*batch_size] = numpy.squeeze(self.tfComputeDescr(miniBatchIdx))
             
        return descr
    
    
    
    def test_on_train(self):
        
        floatX = theano.config.floatX  # @UndefinedVariable
        
        if self.tfErrAndDstDiffOnTrain is None:
            self.setupErrAndDstDiffOnTrain()
        
        n_train_batches = self.n_train_batches
        errs = []
        rawdst = []
        for i in xrange(n_train_batches):
#             if self.macroBatchSize > 0:
#                 self.setMacroBatchData(self.traindataDB,numpy.floor(i / self.macroBatchSize).astype(numpy.int))
#                 miniBatchIdx = numpy.mod(i,self.macroBatchSize)
#             else:
#                 miniBatchIdx = i
            miniBatchIdx = self.dataManager.makeMinibatchAvailable(self.traindataDB,i)
            trainErrors,trainRawDsts = self.tfErrAndDstDiffOnTrain(miniBatchIdx)
            errs.append(trainErrors)
            rawdst.append(trainRawDsts)
        
        print("t {}, l {}, t[0] {}".format(type(errs),len(errs),type(errs[0])))
        print("t {}, l {}, t[0] {}".format(type(rawdst),len(rawdst),type(rawdst[0])))
        
        rawdst = numpy.array(rawdst)
        print("t {}, t[0] {}, s {}".format(type(rawdst),type(rawdst[0]),rawdst.shape))
        
        #thresh = 0.
        thresh = -0.01
        p = (rawdst > thresh).sum()
        n = (rawdst <= thresh).sum()
        print("p {}, n {}, %p {}".format(p,n,100.*p/float(n+p)))
    
    
    def computeTrainDescriptors(self):
        
        # TODO: use self.computeDescriptors(self.traindataDB)  ?
        
        batchSize = self.cfgParams.batch_size
        nSamp = self.n_train_batches * batchSize
        descr = numpy.zeros((nSamp,self.descrNet.cfgParams.outputDim[1]))
                            
        for i in range(self.n_train_batches):
#             if self.macroBatchSize > 0:
#                 self.setMacroBatchData(self.traindataDB,numpy.floor(i / self.macroBatchSize).astype(numpy.int))
#                 miniBatchIdx = numpy.mod(i,self.macroBatchSize)
#             else:
#                 miniBatchIdx = i
            miniBatchIdx = self.dataManager.makeMinibatchAvailable(self.traindataDB,i)   
            descr[i*batchSize:(i+1)*batchSize] = self.tfComputeDescr(miniBatchIdx)
            
        return descr
    
    
    
    def computeValDescriptors(self):
        
        # TODO: use self.computeDescriptors(self.valdataDB)  ?
        
        batchSize = self.cfgParams.batch_size
        nSamp = self.n_val_batches * batchSize
        descr = numpy.zeros((nSamp,self.descrNet.cfgParams.outputDim[1]))
                            
        for i in range(self.n_val_batches):
#             if self.macroBatchSize > 0:
#                 self.setMacroBatchData(self.valdataDB,numpy.floor(i / self.macroBatchSize).astype(numpy.int))
#                 miniBatchIdx = numpy.mod(i,self.macroBatchSize)
#             else:
#                 miniBatchIdx = i
            miniBatchIdx = self.dataManager.makeMinibatchAvailable(self.valdataDB,i)   
            descr[i*batchSize:(i+1)*batchSize] = self.tfComputeDescr(miniBatchIdx)
            
        return descr    
    
    
    
    def computeValDataDistanceMatrix(self):
        
        # TODO: use self.computeDescriptors(self.valdataDB)  ?

        batchSize = self.cfgParams.batch_size
        nSamp = self.n_val_batches * batchSize
        descr = numpy.zeros((nSamp,self.descrNet.cfgParams.outputDim[1]))
                             
        for i in range(self.n_val_batches):
#             if self.macroBatchSize > 0:
#                 self.setMacroBatchData(self.valdataDB,numpy.floor(i / self.macroBatchSize).astype(numpy.int))
#                 miniBatchIdx = numpy.mod(i,self.macroBatchSize)
#             else:
#                 miniBatchIdx = i
            miniBatchIdx = self.dataManager.makeMinibatchAvailable(self.valdataDB,i)
            d = self.tfComputeDescr(miniBatchIdx)
            descr[i*batchSize:(i+1)*batchSize] = d
             
        dst = scipy.spatial.distance.pdist(descr,'euclidean')
        dst = scipy.spatial.distance.squareform(dst) 
                
        return dst
    
    
    
    
    def computeDistanceMatrix(self,test_set):

        batch_size = self.cfgParams.batch_size
        nSamp = test_set.numSamples
        
        descrLen = self.descrNet.cfgParams.outputDim[1]
        descr = numpy.zeros((nSamp,descrLen))

        n_test_batches = nSamp / batch_size
        for i in range(n_test_batches):
#             if self.macroBatchSize > 0:
#                 self.setMacroBatchData(test_set,numpy.floor(i / self.macroBatchSize).astype(numpy.int))
#                 miniBatchIdx = numpy.mod(i,self.macroBatchSize)
#             else:
#                 miniBatchIdx = i
            miniBatchIdx = self.dataManager.makeMinibatchAvailable(test_set,i)                
            d = self.tfComputeDescr(miniBatchIdx)
            descr[i*batch_size:(i+1)*batch_size] = d
             
        print("distances done")
        
        dst = scipy.spatial.distance.pdist(descr,'euclidean')
        dst = scipy.spatial.distance.squareform(dst) 
                
        return dst    
    


    def setupDebugFunctions(self):
        
        #floatX = theano.config.floatX  # @UndefinedVariable
        #batch_size = self.cfgParams.batch_size
#        if self.cfgParams.use_pairs:         
#            batch_idx_size = self.nPairsPerBatch
#        
#         print("compiling compute_train_descr() ... ",end="")
#         givens_train_descr = { self.x: dm.data_x[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] }
#         self.compute_train_descr = theano.function(inputs = [self.tvIndex],
#                                                    outputs = self.descrNet.output,
#                                                    givens = givens_train_descr )
#         print("done.")

        
#         if self.cfgParams.use_pairs:
#             batch_idx_size = self.nPairsPerBatch
#             print("compiling compute_pair_xent() ... ",end="")
#             givens_pair_xent = { self.x: dm.data_x[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] }
#             givens_pair_xent[self.pairIdx] = dm.data_pairIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size] 
#             givens_pair_xent[self.pairLabels] = dm.data_pairLabels[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size] 
#                 
#             self.tfComputePairXent = theano.function(inputs = [self.tvIndex],
#                                                      outputs = self.pair_xent,
#                                                      givens = givens_pair_xent,
#                                                      mode='DebugMode')        
#             print("done.")
        
#         print("compiling compute_pair_dst() ... ",end="")
#         givens_pair_dst = { self.x: self.train_data_x[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size],
#                             self.pairIdx: self.train_data_pairIdx[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] }
#         self.tfComputePairDst = theano.function(inputs = [self.tvIndex],
#                                                 outputs = self.pair_distancesq,
#                                                 givens = givens_pair_dst,
#                                                 mode='DebugMode')
#         print("done.")         

#             print("compiling compute_pair_diff() ... ",end="")
#             givens_pair_diff = { self.x: dm.data_x[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size],
#                                  self.pairIdx: dm.data_pairIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size] }
#             self.tfComputePairDiff = theano.function(inputs = [self.tvIndex],
#                                                      outputs = self.pair_difference,
#                                                      givens = givens_pair_diff,
#                                                      mode='DebugMode')
#             print("done.")    
# 
#             print("compiling compute_pair_diff_idx0() ... ",end="")
#             givens_pair_diff_idx0 = { self.pairIdx: dm.data_pairIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size] }
#             self.tfComputePairDiffIdx0 = theano.function(inputs = [self.tvIndex],
#                                                           outputs = self.pair_diff_idx0,
#                                                           givens = givens_pair_diff_idx0,
#                                                           mode='DebugMode')
#             print("done.")    
#             print("compiling compute_pair_diff_idx1() ... ",end="")
#             givens_pair_diff_idx1 = { self.pairIdx: dm.data_pairIdx[self.tvIndex * batch_idx_size:(self.tvIndex + 1) * batch_idx_size] }
#             self.tfComputePairDiffIdx1 = theano.function(inputs = [self.tvIndex],
#                                                           outputs = self.pair_diff_idx1,
#                                                           givens = givens_pair_diff_idx1,
#                                                           mode='DebugMode')
#             print("done.")
        pass    

    def mineHardNegativeTrainingPairsWithinMiniBatches(self):
        
        dnParams = self.descrNet.cfgParams
        batch_size = self.cfgParams.batch_size
        pairIdx = self.tvPairIdx
        #pairLabels = self.tvPairLabels
        y = self.tvY
        margin = self.pair_neg_margin
        
        
        diff = self.descrNet.output[pairIdx[:,0]] - self.descrNet.output[pairIdx[:,1]]
        dst = T.sum(diff**2,axis=1) / dnParams.outputDim[1]  # divide by number of outputs, such that the max distance is 1
        
        pairLabels = T.eq(y[pairIdx[:,0]],y[pairIdx[:,1]])  #  same class / different class ?
        pair_cost = pairLabels*dst + (1-pairLabels)*T.sqr(T.maximum(0,margin - T.sqrt(dst)))                
        
        # indices for all pairs of vectors in the minibatch
        pidx1,pidx2 = numpy.triu_indices(batch_size, 1) #numpy.mask_indices(batch_size, numpy.triu, 1)
        pidx1 = pidx1.reshape((len(pidx1),1))
        pidx2 = pidx2.reshape((len(pidx2),1))
        comb_pairIdx = numpy.concatenate((pidx1,pidx2),axis=1).astype(numpy.int32)
        
        dm = self.dataManager
        
        if isinstance(self.tvX,list):            
            givens = { tv: data[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] for (tv,data) in zip(self.tvX,dm.tvsData_x) }
        else:
            givens = { self.tvX : dm.tvsData_x[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] }
        givens[self.y] = dm.tvsData_y[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size]
        givens[pairIdx] = comb_pairIdx 
        
        tf = theano.function(inputs=[self.tvIndex],
                            outputs=[pair_cost],
                            givens=givens)
        
        # for every sample get the index of the other sample with which together it forms the most expensive (highest cost) pair 
        nSamp = self.n_train_batches*batch_size
        idx = numpy.zeros(nSamp,dtype=numpy.int32)
        labels = numpy.zeros(nSamp,dtype=numpy.int32)  
        for i in range(self.n_train_batches):  
#             if self.macroBatchSize > 0:
#                 self.setMacroBatchData(self.traindataDB,numpy.floor(i / self.macroBatchSize).astype(numpy.int))
#                 miniBatchIdx = numpy.mod(i,self.macroBatchSize)
#             else:
#                 miniBatchIdx = i
            miniBatchIdx = self.dataManager.makeMinibatchAvailable(self.traindataDB,i)            
            c = tf(miniBatchIdx)
            c = scipy.spatial.distance.squareform(c[0])
            # find the max for each
            offset = i*batch_size
            maxIdx = numpy.argmax(c,axis=0) + offset 
            idx[i*batch_size:(i+1)*batch_size] = maxIdx 
            labels[i*batch_size:(i+1)*batch_size] = self.traindataDB.y[maxIdx] == self.traindataDB.y[i*batch_size:(i+1)*batch_size]
            
            #print(c)
            
        idx = numpy.concatenate((numpy.arange(nSamp,dtype=numpy.int32).reshape(nSamp,1),idx.reshape(nSamp,1)),axis=1)
        
        return idx,labels
        

    def mineHardNegativeTrainingTripletsWithinMiniBatches(self):
        
        dnParams = self.descrNet.cfgParams
        batch_size = self.cfgParams.batch_size
        pairIdx = self.tvPairIdx
        
        #pairLabels = self.pairLabels
        #y = self.y        
        #margin = self.pair_neg_margin
        #rot = T.vector()
 
        # indices for all pairs of vectors in the minibatch
        #pidx1,pidx2 = numpy.mask_indices(batch_size, numpy.triu, 1)
        pidx1,pidx2 = numpy.triu_indices(batch_size, 1)
        #pidx1 = pidx1.reshape((len(pidx1),1))
        #pidx2 = pidx2.reshape((len(pidx2),1))
        comb_pairIdx = numpy.concatenate((pidx1.reshape((len(pidx1),1)),pidx2.reshape((len(pidx2),1))),axis=1).astype(numpy.int32)

        diff = self.descrNet.output[pairIdx[:,0]] - self.descrNet.output[pairIdx[:,1]]
        dst = T.sum(diff**2,axis=1) / dnParams.outputDim[1]  # divide by number of outputs, such that the max distance is 1

        theta = 10.0*numpy.pi/180.0
        #pairLabels = T.eq(y[pairIdx[:,0]],y[pairIdx[:,1]])  #  same class / different class ?
        #pairRotSim = abs(rot[pairIdx[:,0]] - rot[pairIdx[:,1]]) < theta # same (or similar) rotation ?
        #pair_cost = pairLabels*dst + (1-pairLabels)*T.sqr(T.maximum(0,margin - T.sqrt(dst)))                
        
        dm = self.dataManager
         
        if isinstance(self.tvX,list):            
            givens = { tv: data[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] for (tv,data) in zip(self.tvX,dm.tvsData_x) }
        else:
            givens = { self.tvX : dm.tvsData_x[self.tvIndex * batch_size:(self.tvIndex + 1) * batch_size] }
        givens[pairIdx] = comb_pairIdx
        
        tfCalcDists = theano.function(inputs=[self.tvIndex],
                                     outputs=[dst],
                                     givens=givens)
        
        # for every sample get the index of the other sample with which together it forms the most expensive (highest cost) pair
        rots = self.traindataDB.sampleInfo['rots']
        maxRots = self.traindataDB.sampleInfo['maxRots']
        labels = self.traindataDB.y
        nSamp = self.n_train_batches*batch_size
        nNewTripletsPerBatch = batch_size  # one new triplet per sample
        nNewTriplets = self.n_train_batches*nNewTripletsPerBatch        
        #sidx = numpy.zeros((nSamp,1),dtype=numpy.int32)
        #didx = numpy.zeros((nSamp,1),dtype=numpy.int32)
        tripletIdx = numpy.zeros((nNewTriplets,3),dtype=numpy.int32)
        tIdx = 0
        for nBatch in range(self.n_train_batches): 
            
            batchOffset = nBatch*batch_size
            batchLabels = labels[batchOffset:batchOffset+batch_size]  # labels of all sample in this batch
            pidx1Labels = batchLabels[pidx1]  # n*(n-1)/2 x 1
            pidx2Labels = batchLabels[pidx2]  # n*(n-1)/2 x 1
            
            #pairSameClass = labels[pidx1+batchOffset] == labels[pidx2+batchOffset]  # n*(n-1)/2 x 1
            pairSameClass = pidx1Labels == pidx2Labels  # n*(n-1)/2 x 1
            pairSameClassSq = scipy.spatial.distance.squareform(pairSameClass) + numpy.eye(batch_size) # n x n
            
            rotations = rots[batchOffset:batchOffset+batch_size]  # n x 1
            maxRotations = maxRots[batchOffset:batchOffset+batch_size]
            rotations = (rotations / maxRotations) % 1  # normalize rotations to 0..1
            
            pairRotDiff = scipy.spatial.distance.pdist(rotations.reshape(len(rotations),1), 'euclidean')   # n*(n-1)/2 x 1
            pairRotDiff = 0.5 - abs(pairRotDiff - 0.5) 
            pairRotDiff = scipy.spatial.distance.squareform(pairRotDiff)  # n x n
            
#             if self.macroBatchSize > 0:
#                 self.setMacroBatchData(self.traindataDB,numpy.floor(nBatch / self.macroBatchSize).astype(numpy.int))
#                 miniBatchIdx = numpy.mod(nBatch,self.macroBatchSize)
#             else:
#                 miniBatchIdx = nBatch
            miniBatchIdx = self.dataManager.makeMinibatchAvailable(self.traindataDB,nBatch)  
            descr_dst = tfCalcDists(miniBatchIdx) 
            descr_dst = scipy.spatial.distance.squareform(descr_dst[0])  # n x n

            # v1            
            # for each class individually,
            #   for each sample find the most distant sample that is still similar in pose
            
            # v2
            # for each class individually,
            #   for each sample find the most              

            #tripletRotDiff = scipy.spatial.distance.pdist(pairRotDiff, 'euclidean') # n*(n-1)/2 x n
            tripletRotDiff = pairRotDiff[pidx1] - pairRotDiff[pidx2]  # n*(n-1)/2 x n,   because len(pidx1) = n*(n-1)/2,  if negative than pair 1 is more similar 
            tripletDstDiff = descr_dst[pidx1] - descr_dst[pidx2]    # should be negative, to be okay (i.e. dst of same is smaller than dst of diff)
            
            nTriplets = tripletDstDiff.shape[0]    # TODO: assert that this is batch_size * (batch_size-1) / 2
            #batch_size = tripletDst.shape[1]  # TODO: could use an assert here
            assert tripletDstDiff.shape[1] == batch_size 
            assert nTriplets == batch_size*(batch_size-1)/2
            
            tripletCost = numpy.zeros((nTriplets,batch_size))
            
            #tripletCost = tripletDst*((tripletRotDiff > 0) or ) # idx1issamebutidx2isnotsameclass, i.e., isOtherClassTriplet (if both idx1 and idx2 are other class i dont care)
            for j in range(nTriplets): # over all triplets
                for i in range(batch_size): # over all samples (root sample of the triplet)
                    psc1 = pairSameClassSq[i,pidx1[j]]
                    psc2 = pairSameClassSq[i,pidx2[j]]
                    if psc1 and psc2:
                        # both same class, look at rotation angle difference
                        tripletCost[j,i] = max(0, - tripletDstDiff[j,i] * numpy.sign(tripletRotDiff[j,i]))
                        # if tripletRotDiff is positive sample2 is more similar to the root sample than sample1
                        #   then tripletDstDiff should be positive, to be okay (sample2 closer than sample1)
                    elif (not psc1) and (not psc1):
                        # both are not from the same class
                        # this is not an interesting triplet 
                        # (these are actually many. we should not have to test them, but this would mean testing the classes individually .... maybe we should do that)
                        tripletCost[j,i] = 0  
                    elif psc1:
                        # sample 1 is from the same class as the root sample i
                        # thus it must be closer than sample 2 -> i.e. the dstdiff should be negative 
                        tripletCost[j,i] = max(0.0,tripletDstDiff[j,i])
                    else:
                        # sample 2 is from the same class as the root sample i -> dstdiff should be positive
                        tripletCost[j,i] = max(0.0,-tripletDstDiff[j,i])
                        
            # find for each root sample the triplet that costs most
            maxCostIdx = numpy.argmax(tripletCost,axis=0)
            maxCost = tripletCost[maxCostIdx,numpy.arange(batch_size)]

            # find corresponding sample indices for those most costly triplets
            idx0 = numpy.arange(batch_size)
            idx1 = pidx1[maxCostIdx]
            idx2 = pidx2[maxCostIdx]
            
            # now we have triplets, but idx1 and idx2 might need to be exchanged
            for i in range(nNewTripletsPerBatch): 
                
                # debug. just for the logics. if this assert is wrong than we have a serious flaw
                assert (tripletRotDiff[maxCostIdx[i],i] > 0)  == (pairRotDiff[i,idx1[i]] > pairRotDiff[i,idx2[i]])
                
                if maxCost[i] == 0:
                    # we shouldn't make a new triplet here, but that's not so easy ...
                    # ... nice thing would be to take another triplet with high cost for a different sample
                    idx1[i] = idx0[i]
                    idx2[i] = idx0[i]
                    
                else:
                
                    if (not pairSameClassSq[idx0[i],idx1[i]]) or (tripletRotDiff[maxCostIdx[i],i] > 0):  # or (pairRotDiff[i,idx1[i]] > pairRotDiff[i,idx2[i]])
                        h = idx1[i]
                        idx1[i] = idx2[i]
                        idx2[i] = h
                         
            idx = numpy.concatenate((idx0.reshape((-1,1)),idx1.reshape((-1,1)),idx2.reshape((-1,1))),axis=1)
            
            tIdx = nBatch*nNewTripletsPerBatch   
            tripletIdx[tIdx:tIdx+nNewTripletsPerBatch,:] = idx
             
#             # find the max for each
#             #maxIdx = numpy.argmax(c,axis=0) + offset
#             maxIdx = numpy.argmax(d * sameRot ,axis=0) 
#              
#             offset = nBatch*batch_size
#             sidx[nBatch*batch_size:(nBatch+1)*batch_size] = maxIdx + offset 
#             
#             #print(c)
            
        
        return tripletIdx


