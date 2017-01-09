'''
Created on May 4, 2015

@author: wohlhart
'''
import theano
import theano.tensor as T
import numpy
import cPickle
import os


class BatchedDataset(object):
    '''
    Struct to hold a stacks of image data and labels indicating target class, pairs of similar and dissimilar samples and triplets of sim/dissim
    '''
    def __init__(self,x=None,y=None,yr=None,sampleInfo=None,batchSize=None):
        self.x = x
        self.y = y
        self.yr = yr  # regression
        self.sampleInfo = sampleInfo  #  additional stuff like pose, ...
        self.batchSize = batchSize
        
        
    def append(self,other):
        """ 
        concatenate the other BatchedDataset to this one
        """
        if self.batchSize != other.batchSize:
            raise ValueError("Cannot concatenate datasets with differing batchSize {} != {}".format(self.batchSize,other.batchSize)) 
        
        if (self.x is not None) and (other.x is not None):
            if isinstance(self.x,list):
                self.x = [numpy.concatenate((sx,ox),axis=0) for sx,ox in zip(self.x,other.x)]
            else:
                self.x = numpy.concatenate((self.x,other.x),axis=0)
        elif (other.x is not None):
            self.x = other.x

        if (self.y is not None) and (other.y is not None):
            self.y = numpy.concatenate((self.y,other.y),axis=0)
        elif (other.y is not None):
            self.y = other.y

        if (self.yr is not None) and (other.yr is not None):
            self.yr = numpy.concatenate((self.yr,other.yr),axis=0)
        elif (other.yr is not None):
            self.yr = other.yr
        
        # ... in derived classes also deal with/append sampleInfo
      
    def subset(self,startBatchIdx,endBatchIdx):
        startIdx = startBatchIdx * self.batchSize
        endIdx = endBatchIdx * self.batchSize
        if isinstance(self.x,list):
            dx = [x[startIdx:endIdx] for x in self.x]
        else:  
            dx = self.x[startIdx:endIdx]
        dy = self.y[startIdx:endIdx] if (self.y is not None) else None
        dyr = self.yr[startIdx:endIdx] if (self.yr is not None) else None
        return BatchedDataset(dx,dy,dyr,self.sampleInfo,self.batchSize)
    
    @property
    def dataSizeNBytes(self):
        if isinstance(self.x, list):
            return sum([x.nbytes for x in self.x])
        else:
            return self.x.nbytes
    
    @property
    def xShape(self):
        if isinstance(self.x, list):
            return [x.shape for x in self.x]
        else:
            return self.x.shape

    @property    
    def numSamples(self):        
        # TODO: numSamples should not just be x.shape[0], because the array may be padded with zeros (or garbage)
        #         however, for now it's like that all over the code, so changing will need a lot of refactoring
        if isinstance(self.x, list):
            return self.x[0].shape[0] # assuming they are all the same. they should. we could assert this here
        else:
            return self.x.shape[0]
        
    @classmethod
    def loadFromPkl(cls,pklFileName):
        
        with open(pklFileName, 'rb') as f:
            data_set = cPickle.load(f)
      
        if isinstance(data_set.x,str):
            with open(os.path.join(os.path.dirname(pklFileName),data_set.x),'rb') as f:
                data = numpy.load(f)
                keys = sorted(data.keys())
                print("loading from keys: {}".format([x for x in keys]))
                if len(keys) > 1:
                    data_set.x = [data[key] for key in keys]
                else:
                    data_set.x = data[keys[0]]
    
        return data_set
    
    
    def saveToPkl(self,pklFileName):
        
        dataSize = self.dataSizeNBytes
        saveXseparately = dataSize > 2**31   # cannot pickle arrays larger than 2**32
        if saveXseparately:
            xFilename = pklFileName + '_x.npz'
            print("saving data.x to extra file: '{}'".format(xFilename))
            with open(xFilename,'wb') as f:
                if isinstance(self.x,list):
                    # make sure the name is such that when sorting the keys they give the correct order
                    numDigits = int(numpy.floor(numpy.log10(len(self.x)-1)))+1 if len(self.x) > 1 else 1
                    fmtstr = "x_{{:0{}d}}".format(numDigits)  
                    names = [fmtstr.format(i) for i in xrange(len(self.x))]
                    numpy.savez(f,**dict(zip(names,self.x)))
                else:           
                    numpy.savez(f,self.x)
            tmpX = self.x 
            self.x = os.path.basename(xFilename)
 
        f = open(pklFileName, 'wb')
        cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()          
        
        if saveXseparately:
            self.x = tmpX

    
class BatchedTrainDataset(BatchedDataset):
    '''
    Data plus all kinds of indices and labels
    
    TODO: move this out into special applications; the base class here should probably not need to know about tripletIdx,...
    '''
    
    def __init__(self, x, y=None, yr=None, sampleInfo=None, pairIdx=None, pairLabels=None, tripletIdx=None, tripletPoolIdx=None,
                 nPairsPerBatch=None, nTripletsPerBatch=None, nTripletPoolsPerBatch=0,
                 negTripletPoolSizeVal=0, posTripletPoolSizeVal=0, batchSize=None):
        super(BatchedTrainDataset,self).__init__(x,y,yr,sampleInfo,batchSize)
        self.pairIdx = pairIdx
        self.pairLabels = pairLabels
        self.nPairsPerBatch = nPairsPerBatch
        self.tripletIdx = tripletIdx
        self.nTripletsPerBatch = nTripletsPerBatch
        self.tripletPoolIdx = tripletPoolIdx
        self.nTripletPoolsPerBatch = nTripletPoolsPerBatch
        self.negTripletPoolSizeVal = negTripletPoolSizeVal
        self.posTripletPoolSizeVal = posTripletPoolSizeVal
         
        
    def append(self,other):
        """ concatenate the other BatchedTrainDataset to this one
        """
        if self.batchSize != other.batchSize:
            raise ValueError("Cannot concatenate datasets with differing batchSize {} != {}".format(self.batchSize,other.batchSize)) 
        if self.nPairsPerBatch != other.nPairsPerBatch:
            raise ValueError("Cannot concatenate datasets with differing nPairsPerBatch {} != {}".format(self.nPairsPerBatch,other.nPairsPerBatch)) 
        if self.nTripletsPerBatch != other.nTripletsPerBatch:
            raise ValueError("Cannot concatenate datasets with differing nTripletsPerBatch {} != {}".format(self.nTripletsPerBatch,other.nTripletsPerBatch)) 
        
        #print("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
        #print("  concatenating datasets ")
        
        super(BatchedTrainDataset,self).append(other)
        
        if (self.pairIdx is not None) and (other.pairIdx is not None):
            self.pairIdx = numpy.concatenate((self.pairIdx,other.pairIdx),axis=0)
        elif (other.pairIdx is not None):
            self.pairIdx = other.pairIdx      

        if (self.pairLabels is not None) and (other.pairLabels is not None):
            self.pairLabels = numpy.concatenate((self.pairLabels,other.pairLabels),axis=0)
        elif (other.pairIdx is not None):
            self.pairLabels = other.pairLabels
            
        if (self.tripletIdx is not None) and (other.tripletIdx is not None):
            self.tripletIdx = numpy.concatenate((self.tripletIdx,other.tripletIdx),axis=0)
        elif other.tripletIdx is not None:
                self.tripletIdx = other.tripletIdx 

        # ... in derived classes also deal with/append sampleInfo


    def extendSetBySubset(self,batchIdx):
        # batchIdx: which batches to replicate

        sampIdxInBatch = numpy.tile(numpy.arange(self.batchSize),len(batchIdx))
        batchOffset = numpy.repeat(batchIdx,self.batchSize)*self.batchSize
        sampIdx = batchOffset + sampIdxInBatch

        if isinstance(self.x,list):
            for i,x in enumerate(self.x):
                self.x[i] = numpy.concatenate((x,x[sampIdx]),axis=0)
        else:
            self.x = numpy.concatenate((self.x,self.x[sampIdx]),axis=0)
        
        self.y = numpy.concatenate((self.y,self.y[sampIdx]),axis=0)

        if self.yr is not None:         
            self.yr = numpy.concatenate((self.yr,self.yr[batchIdx]),axis=0)

        if self.pairIdx is not None:         
            self.pairIdx = numpy.concatenate((self.pairIdx,self.pairIdx[sampIdx]),axis=0)
            self.pairLabels = numpy.concatenate((self.pairLabels,self.pairLabels[sampIdx]),axis=0)
            
        if self.tripletIdx is not None:
            self.tripletIdx = numpy.concatenate((self.tripletIdx,self.tripletIdx[sampIdx]),axis=0)


    def subset(self,startBatchIdx,endBatchIdx):
        startIdx = startBatchIdx * self.batchSize
        endIdx = endBatchIdx * self.batchSize
        if isinstance(self.x,list): 
            x = [dx[startIdx:endIdx] for dx in self.x]
        else:
            x = self.x[startIdx:endIdx]  
        y = self.y[startIdx:endIdx] if (self.y is not None) else None
        yr = self.yr[startIdx:endIdx] if (self.yr is not None) else None
        
        if self.pairIdx is not None:
            pairsStart = startBatchIdx * self.nPairsPerBatch
            pairsEnd = endBatchIdx * self.nPairsPerBatch
            pairIdx = self.pairIdx[pairsStart:pairsEnd]
            pairLabels = self.pairLabels[pairsStart:pairsEnd] 
        else:  
            pairIdx = None
            pairLabels = None
        
        if self.tripletIdx is not None:
            tripletsStart = startBatchIdx * self.nTripletsPerBatch
            tripletsEnd = endBatchIdx * self.nTripletsPerBatch
            tripletIdx = self.tripletIdx[tripletsStart:tripletsEnd]
        else:  
            tripletIdx = None

        if self.tripletPoolIdx is not None:
            tripletPoolsStart = startBatchIdx * self.nTripletPoolsPerBatch 
            tripletPoolsEnd = endBatchIdx * self.nTripletPoolsPerBatch 
            tripletPoolIdx = self.tripletPoolIdx[tripletPoolsStart:tripletPoolsEnd]
        else:  
            tripletPoolIdx = None
            
        sampleInfo = self.sampleInfo # deal with this in derived versions
        return BatchedTrainDataset(x,y,yr,sampleInfo,pairIdx,pairLabels,tripletIdx,tripletPoolIdx,
                                   self.nPairsPerBatch,self.nTripletsPerBatch,self.nTripletPoolsPerBatch,
                                   self.negTripletPoolSizeVal,self.posTripletPoolSizeVal,self.batchSize)


class DataManagerConfig(object):
    def __init__(self,batch_size,use_labels,use_regtargets,use_pairs,use_triplets,use_tripletPools):
        self.batch_size = batch_size
        self.use_labels = use_labels
        self.use_regtargets = use_regtargets
        self.use_pairs = use_pairs
        self.use_triplets = use_triplets
        self.use_tripletPools = use_tripletPools
        
    
class DataManager(object):
    
    def __init__(self,cfg):
        '''
        @type cfg: DataManagerConfig
        '''
        self.currentDataSet = None           
        self.cfg = cfg
     

    def setupVariables(self,data_set=None,doBorrow=True):
        '''
        @type data_set: BatchedDataset
        '''

        # TODO: move all this specific stuff of pairs and triplets into a derived class
        
        floatX = theano.config.floatX  # @UndefinedVariable
        
        if data_set is None:
            # make a small dummy dataset
            nSamp = 1
            nPairsPerBatch = nSamp*1
            nTripletsPerBatch = nSamp*1
            nTripletPoolsPerBatch = nSamp*1
            negTripletPoolSizeVal = 1
            posTripletPoolSizeVal = 1
            batchSize = 1
            x = numpy.array((nSamp,1,1,1),dtype=floatX)  # TODO: how to know if we should instantiate a list of xs here?
            y = numpy.array((nSamp,1),dtype=numpy.int32)
            yr = numpy.array((nSamp,1),dtype=floatX)
            pairIdx = numpy.array((nSamp,2),dtype=numpy.int32)
            pairLabels = numpy.array((nPairsPerBatch,1),dtype=numpy.int32)
            tripletIdx = numpy.array((nSamp,3),dtype=numpy.int32)
            tripletPoolIdx = numpy.array((nSamp,3),dtype=numpy.int32)
            ds = BatchedTrainDataset(x, y, yr, [], pairIdx, pairLabels, tripletIdx, tripletPoolIdx, 
                                           nPairsPerBatch, nTripletsPerBatch, 
                                           nTripletPoolsPerBatch, negTripletPoolSizeVal, posTripletPoolSizeVal, 
                                           batchSize)
        else:
            ds = data_set
            
        # shared vars for data
        if isinstance(ds.x,list):
            self.tvsData_x = [theano.shared(x, name='data_x_{}'.format(i), borrow=doBorrow) for i,x in enumerate(ds.x)]
        else:
            self.tvsData_x = theano.shared(ds.x, name='data_x', borrow=doBorrow)
         
        #if self.cfgParams.use_labels:
        if self.cfg.use_labels and ds.y is not None:
            self.tvsData_y = theano.shared(ds.y, name='data_y', borrow=doBorrow)
             
        if self.cfg.use_regtargets:
            self.tvsData_yr = theano.shared(ds.yr, name='data_yr', borrow=doBorrow)
 
        if self.cfg.use_pairs:            
            self.tvsData_pairIdx = theano.shared(ds.pairIdx, name='data_pairidx', borrow=doBorrow)
            self.tvsData_pairLabels = theano.shared(ds.pairLabels, name='data_pairlabels', borrow=doBorrow)
            self.nPairsPerBatch = ds.nPairsPerBatch
 
        if self.cfg.use_triplets:
            self.tvsData_tripletIdx = theano.shared(ds.tripletIdx, name='data_tripletidx', borrow=doBorrow)
            self.nTripletsPerBatch = ds.nTripletsPerBatch
            
        if self.cfg.use_tripletPools:
            self.tvsData_tripletPoolIdx = theano.shared(ds.tripletPoolIdx, name='data_tripletpoolidx', borrow=doBorrow)
            self.nTripletPoolsPerBatch = ds.nTripletPoolsPerBatch
            self.negTripletPoolSizeVal = ds.negTripletPoolSize
            self.posTripletPoolSizeVal = ds.posTripletPoolSize
            
            
        if data_set is None:
            self.prepareForNewData()
            
        self.currentDataSet = data_set
            
            
    def makeMinibatchAvailable(self,data_set,minibatch_index):
        
        if (self.currentDataSet is not data_set):
            # ------ set values ----------------
            doSetLabels = self.cfgParams.use_labels and (data_set.y is not None)
            doSetTargets = self.cfgParams.use_regtargets and (self.tvsData_yr is not None)
            doSetPairs = self.cfgParams.use_pairs and hasattr(data_set,'pairIdx') and (data_set.pairIdx is not None)
            doSetTriplets = self.cfgParams.use_triplets and hasattr(data_set,'tripletIdx') and (data_set.tripletIdx is not None)
            doSetTripletPools = self.cfgParams.use_tripletPools and hasattr(data_set,'tripletPoolIdx') and (data_set.tripletPoolIdx is not None)

            doBorrow = True     
            if isinstance(self.tvsData_x,list):         
                for (dx,dsx) in zip(self.tvsData_x,data_set.x): 
                    dx.set_value(dsx,borrow=doBorrow)
            else:
                self.tvsData_x.set_value(data_set.x,borrow=doBorrow)
                
            if doSetLabels:
                self.tvsData_y.set_value(data_set.y,borrow=doBorrow)
                
            if doSetTargets:
                self.tvsData_yr.set_value(data_set.yr,borrow=doBorrow)
                    
            if doSetPairs:
                assert self.nPairsPerBatch < 0 or self.nPairsPerBatch == data_set.nPairsPerBatch
                self.tvsData_pairIdx.set_value(data_set.pairIdx,borrow=doBorrow)
                self.tvsData_pairLabels.set_value(data_set.pairLabels,borrow=doBorrow)
                self.nPairsPerBatch = data_set.nPairsPerBatch
            
            if doSetTriplets:
                assert self.nTripletsPerBatch < 0 or self.nTripletsPerBatch == data_set.nTripletsPerBatch
                self.tvsData_tripletIdx.set_value(data_set.tripletIdx,borrow=doBorrow)
                self.nTripletsPerBatch = data_set.nTripletsPerBatch

            if doSetTripletPools:
                assert self.nTripletPoolsPerBatch < 0 or self.nTripletPoolsPerBatch == data_set.nTripletPoolsPerBatch
                self.tvsData_tripletPoolIdx.set_value(data_set.tripletPoolIdx,borrow=doBorrow)
                self.nTripletPoolsPerBatch = data_set.nTripletPoolsPerBatch
                            
            self.currentDataSet = data_set
        
        #else:
        #    print("DataManager: setting same data -> nothing to do")
        
        return minibatch_index
    
    
    def prepareForNewData(self):
        '''
        Reset Idx sizes such that new data with different sizes than before are allowed
        
        Call this if you want to makeMinibatchAvailable on a new dataset, 
           with data block dimensions different from what was used before  
        '''
        self.nPairsPerBatch = -1
        self.nTripletsPerBatch = -1
        self.nTripletPoolsPerBatch = -1
        
            
class MacroBatchManagerConfig(DataManagerConfig):
    def __init__(self,batch_size=0,use_labels=False,use_regtargets=False,use_pairs=False,use_triplets=False,use_tripletPools=False):
        super(MacroBatchManagerConfig,self).__init__(batch_size,use_labels,use_regtargets,use_pairs,use_triplets,use_tripletPools)
    
    
class MacroBatchManager(DataManager):

    def __init__(self,cfg,macroBatchSize,acceptPartialMacroBatches=False):
        '''
        @type cfg: MacroBatchManagerConfig
        '''
        super(MacroBatchManager, self).__init__(cfg)
        self.currentDataSet = None           
        self.currentMacroBatchIdx = -1
        self.macroBatchSize = macroBatchSize 
        self.acceptPartialMacroBatches = acceptPartialMacroBatches
        
        
    def setupVariables(self,data_set=None,doBorrow=True):        
        
        if self.acceptPartialMacroBatches:
            doBorrow = False # dont borrow if partial macrobatches are used
            
        ds = data_set.subset(0,self.macroBatchSize) if data_set else None
            
        super(MacroBatchManager,self).setupVariables(ds,doBorrow)
                
        self.currentDataSet = data_set   
        self.currentMacroBatchIdx = 0           
        
        
    def makeMinibatchAvailable(self,data_set,minibatch_index):
        macroBatchIdx = numpy.floor(minibatch_index / self.macroBatchSize).astype(numpy.int)
        
        if (self.currentDataSet is not data_set) or (self.currentMacroBatchIdx != macroBatchIdx):
            
            if (self.currentDataSet is data_set):
                print("setting macroBatch {} on same data".format(macroBatchIdx))
            else:
                print("setting macroBatch {} on new data".format(macroBatchIdx))
                
            batch_size = self.cfg.batch_size
            numSamples = data_set.numSamples
            numMiniBatches = numSamples/batch_size   
            macroBatchNSamples = self.macroBatchSize*batch_size
            
            # ------ set values ----------------
            doSetLabels = self.cfg.use_labels and (data_set.y is not None)
            doSetTargets = self.cfg.use_regtargets and (data_set.yr is not None)
            doSetPairs = self.cfg.use_pairs and hasattr(data_set,'pairIdx') and (data_set.pairIdx is not None)
            doSetTriplets = self.cfg.use_triplets and hasattr(data_set,'tripletIdx') and (data_set.tripletIdx is not None)
            doSetTripletPools = self.cfg.use_tripletPools and hasattr(data_set,'tripletPoolIdx') and (data_set.tripletPoolIdx is not None)

            mbS = macroBatchIdx*macroBatchNSamples
            mbE = (macroBatchIdx+1)*macroBatchNSamples
            
            if doSetPairs:
                macroBatchNPairs = data_set.nPairsPerBatch * self.macroBatchSize
                mbSP = macroBatchIdx*macroBatchNPairs
                mbEP = (macroBatchIdx+1)*macroBatchNPairs
            if doSetTriplets:    
                macroBatchNTriplets = data_set.nTripletsPerBatch * self.macroBatchSize
                mbST = macroBatchIdx*macroBatchNTriplets
                mbET = (macroBatchIdx+1)*macroBatchNTriplets
            if doSetTripletPools:
                macroBatchNTripletPools = data_set.nTripletPoolsPerBatch * self.macroBatchSize
                mbSTP = macroBatchIdx*macroBatchNTripletPools
                mbETP = (macroBatchIdx+1)*macroBatchNTripletPools
            

            assert (mbE <= numSamples) or self.acceptPartialMacroBatches, "trying to set a partial macrobatch, but the MacroBatchManager was not configured for that"
            
            if mbE <= numSamples:
                
                doBorrow = not self.acceptPartialMacroBatches
                
                if isinstance(self.tvsData_x,list):
                    for (dx,dsx) in zip (self.tvsData_x,data_set.x):
                        dx.set_value(dsx[mbS:mbE],borrow=doBorrow)
                else:
                    self.tvsData_x.set_value(data_set.x[mbS:mbE],borrow=doBorrow)
                print("set samples {}:{}".format(mbS,mbE))
                
                if doSetLabels:
                    print("...... set labels")
                    self.tvsData_y.set_value(data_set.y[mbS:mbE],borrow=doBorrow)
                    
                if doSetTargets:
                    print("...... set targets")
                    self.tvsData_yr.set_value(data_set.yr[mbS:mbE],borrow=doBorrow)
                    
                if doSetPairs:
                    print("...... set pairs ({}:{})".format(mbSP,mbEP))
                    assert self.nPairsPerBatch < 0 or self.nPairsPerBatch == data_set.nPairsPerBatch, "{} != {}".format(self.nPairsPerBatch, data_set.nPairsPerBatch)
                    self.tvsData_pairIdx.set_value(data_set.pairIdx[mbSP:mbEP],borrow=doBorrow)
                    self.tvsData_pairLabels.set_value(data_set.pairLabels[mbSP:mbEP],borrow=doBorrow)
                    self.nPairsPerBatch = data_set.nPairsPerBatch
                
                if doSetTriplets:
                    print("...... set triplets ({}:{}) # tripl per batch: {},{}".format(mbST,mbET,self.nTripletsPerBatch,data_set.nTripletsPerBatch))
                    assert self.nTripletsPerBatch < 0 or self.nTripletsPerBatch == data_set.nTripletsPerBatch, "{} != {}".format(self.nTripletsPerBatch, data_set.nTripletsPerBatch)
                    self.tvsData_tripletIdx.set_value(data_set.tripletIdx[mbST:mbET],borrow=doBorrow)
                    self.nTripletsPerBatch = data_set.nTripletsPerBatch
                    
                if doSetTripletPools:
                    print("...... set triplet pools ({}:{})".format(mbSTP,mbETP))
                    assert self.nTripletPoolsPerBatch < 0 or self.nTripletPoolsPerBatch == data_set.nTripletPoolsPerBatch, "{} != {}".format(self.nTripletPoolsPerBatch, data_set.nTripletPoolsPerBatch)
                    self.tvsData_tripletPoolIdx.set_value(data_set.tripletPoolIdx[mbSTP:mbETP],borrow=doBorrow)
                    self.nTripletPoolsPerBatch = data_set.nTripletPoolsPerBatch
                            
            else:
                assert mbS < numSamples, "macroBatchIdx {} beyond limit {} ({},{})".format(macroBatchIdx,int(numpy.ceil(numMiniBatches/float(self.macroBatchSize))),numMiniBatches,self.macroBatchSize)                 
                
                # trying to set a macro batch that is not full (a partial macrobatch)
                #print("mbE = {}  before".format(mbE))

                mbE = numSamples
                
                #print("mbS = {}".format(mbS))
                #print("mbE = {}".format(mbE))
                #print("data_set.x[mbS:mbE].shape = {}".format(data_set.x[mbS:mbE].shape))
                
                if isinstance(data_set.x,list):
                    dsx = [x[mbS:mbE] for x in data_set.x]
                else:
                    dsx = data_set.x[mbS:mbE]
                self.tfSetMacroBatchSubsetData(dsx)
                
                if doSetLabels:
                    self.tfSetMacroBatchSubsetY (data_set.y[mbS:mbE])

                if doSetTargets:
                    self.tfSetMacroBatchSubsetYR (data_set.yr[mbS:mbE])

                if doSetPairs:
                    mbEP = data_set.pairIdx.shape[0]
                    #print("mbSP,mbEP {},{}".format(mbSP,mbEP))
                    self.tfSetMacroBatchSubsetPairs(data_set.pairIdx[mbSP:mbEP],data_set.pairLabels[mbSP:mbEP])

                if doSetTriplets:
                    assert self.nTripletsPerBatch == data_set.nTripletsPerBatch, "{} != {}".format(self.nTripletsPerBatch, data_set.nTripletsPerBatch)
                    mbET = data_set.tripletIdx.shape[0]
                    #print("mbST,mbET {},{}".format(mbST,mbET))
                    self.tfSetMacroBatchSubsetTriplets(data_set.tripletIdx[mbST:mbET])

                if doSetTripletPools:
                    assert self.nTripletPoolsPerBatch == data_set.nTripletPoolsPerBatch, "{} != {}".format(self.nTripletPoolsPerBatch, data_set.nTripletPoolsPerBatch) 
                    mbETP = data_set.tripletPoolIdx.shape[0]
                    #print("mbSTP,mbETP {},{}".format(mbSTP,mbETP))
                    self.tfSetMacroBatchSubsetTripletPools(data_set.tripletPoolIdx[mbSTP:mbETP])
                            
            self.currentDataSet = data_set
            self.currentMacroBatchIdx = macroBatchIdx
            
        #else:
        #    print("MacroBatchManager: setting same data -> nothing to do")

    
        miniBatchIdx = numpy.mod(minibatch_index,self.macroBatchSize)
        return miniBatchIdx
    
    
    def setupSetMacroBatchSubset(self):
        
        if isinstance(self.tvsData_x,list):
            data_block = [T.tensor4('data_block_{}'.format(i)) for i in range(len(self.tvsData_x))]
            data_updates = [(dx,T.set_subtensor(dx[:db.shape[0]], db)) for (dx,db) in zip(self.tvsData_x,data_block)]
        else:
            data_block = T.tensor4('data_block')
            data_updates = [(self.tvsData_x, T.set_subtensor(self.tvsData_x[:data_block.shape[0]], data_block))]
        self.tfSetMacroBatchSubsetData = theano.function(inputs=[data_block],updates=data_updates)
        
        if self.cfgParams.use_labels:
            y_block = T.ivector('y_block')
            y_updates = [(self.tvsData_y, T.set_subtensor(self.tvsData_y[:y_block.shape[0]], y_block))]
            self.tfSetMacroBatchSubsetY = theano.function(inputs=[y_block],updates=y_updates)
        
        if self.cfgParams.use_regtargets:
            yr_block = T.vector('yr_block')
            yr_updates = [(self.tvsData_yr, T.set_subtensor(self.tvsData_yr[:yr_block.shape[0]], yr_block))]
            self.tfSetMacroBatchSubsetYR = theano.function(inputs=[yr_block],updates=yr_updates)
                
        if self.cfgParams.use_pairs:
            pairIdx_block = T.imatrix('pairIdx_block')
            pairLabels_block = T.ivector('pairLabels_block')
            pair_updates = [( self.tvsData_pairIdx, T.set_subtensor(self.tvsData_pairIdx[:pairIdx_block.shape[0]], pairIdx_block) ),
                            ( self.tvsData_pairLabels, T.set_subtensor(self.tvsData_pairLabels[:pairLabels_block.shape[0]], pairLabels_block) )]
            self.tfSetMacroBatchSubsetPairs = theano.function(inputs=[pairIdx_block,pairLabels_block],updates=pair_updates)

        if self.cfgParams.use_triplets:
            tripletIdx_block = T.imatrix('tripletIdx_block')
            triplets_updates = [ (self.tvsData_tripletIdx, T.set_subtensor(self.tvsData_tripletIdx[:tripletIdx_block.shape[0]], tripletIdx_block)) ]
            self.tfSetMacroBatchSubsetTriplets = theano.function(inputs=[tripletIdx_block],updates=triplets_updates)
        
        if self.cfgParams.use_tripletPools:            
            tripletPoolIdx_block = T.imatrix('tripletPoolIdx_block')
            tripletPools_updates = [ (self.tvsData_tripletPoolIdx, T.set_subtensor(self.tvsData_tripletPoolIdx[:tripletPoolIdx_block.shape[0]], tripletPoolIdx_block)) ]
            self.tfSetMacroBatchSubsetTripletPools = theano.function(inputs=[tripletPoolIdx_block],updates=tripletPools_updates)


