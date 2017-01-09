'''
Created on Jun 2, 2015

@author: wohlhart
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # @UnresolvedImport @UnusedImport

import sys, os, time, numpy, cv2, scipy
import theano

import logging
from tnetcore.data import BatchedDataset
from util.misc import nowPidStr
from util.vis import montageLinemodTrainSamplesAndTmpls

    
def checkDataset(train_set):
    
    batchSize = train_set.batchSize
    numSamples = train_set.numSamples
    nBatches = numSamples / batchSize
    assert nBatches*batchSize == numSamples, "number of samples {} not divisible by batchSize {}".format(numSamples,batchSize) 
    
    nClasses = len(scipy.setdiff1d(numpy.unique(train_set.y),numpy.array([-1.])))
    print("nClasses {}".format(nClasses))
    
    nTripletsPerBatch = train_set.nTripletsPerBatch
    si = train_set.sampleInfo
    tmplStartIdx = si['tmplBatchDataStartIdx']
    sampIdx = si['sampIdx'] # number of the sample in original per-class sequence
    tmplRots = si['tmplRots']
    trainRots = si['trainRots']
    #nTrainPerSeq = si['nTrainPerSeq']
    zRotInv = si['zRotInv']
    print("numSamples {}".format(numSamples))
    print("batchSize {}".format(batchSize))    
    print("nBatches {}".format(nBatches))
    

    #print("train_set.y\n {}".format(train_set.y.reshape((batchSize,nBatches))))
    print("train_set.y shape {}".format(train_set.y.shape))
    print("numValidSamples {}".format(numpy.sum(train_set.y>=0)))
    #print("tmplStartIdx\n {}".format(tmplStartIdx))
    print("sampIdx\n {}".format(sampIdx))
    
    
    for nBatch in xrange(nBatches):
        for i in xrange(nTripletsPerBatch):
            tIdx = nBatch*nTripletsPerBatch+i
            idx = train_set.tripletIdx[tIdx,:]
            
            # check if idx0 is in the training sample area
            if idx[0] >= tmplStartIdx[nBatch,0]:
                print("ERROR: first index must be train sample but {} >= {}".format(numpy.max(idx[0]),tmplStartIdx[nBatch,0]))
            # check if idx1,idx2 are in the template sample area
            if idx[1] < tmplStartIdx[nBatch,0]:
                print("ERROR: second index must be template sample but {} < {}".format(numpy.max(idx[1]),tmplStartIdx[nBatch,0]))
            if idx[2] < tmplStartIdx[nBatch,0]:
                print("ERROR: third index must be template sample but {} < {}".format(numpy.max(idx[2]),tmplStartIdx[nBatch,0]))
                
            idx = numpy.copy(idx)
            idx = idx + nBatch*batchSize  #** it is now a within batch idx. so to index into the whole dataset add offset
               
            # check if idx0 and idx1 are same class
            l0 = train_set.y[idx[0]] 
            l1 = train_set.y[idx[1]]
            l2 = train_set.y[idx[2]]
            if l0 != l1:
                print("ERROR: l0 != l1")
            else:
                # check if idx2 is also the same
                if (l0 == l2): # and if yes, if the rotation of the second is bigger than the first
                    rot0 = trainRots[l0,sampIdx[idx[0]]]
                    rot1 = tmplRots[l0,sampIdx[idx[1]]]
                    rot2 = tmplRots[l0,sampIdx[idx[2]]]
                    sim1 = numpy.dot(rot0,rot1)
                    sim2 = numpy.dot(rot0,rot2)
                    if zRotInv[l0] == 2:
                        sim1 = numpy.maximum(sim1,numpy.dot(rot0*numpy.array([-1,-1,1]),rot1))
                        sim2 = numpy.maximum(sim2,numpy.dot(rot0*numpy.array([-1,-1,1]),rot2))                    
                    if sim1 < sim2:
                        print("ERROR: s2 is more similar to s0 than s1 !!")
                        print("   idx[0] = {}, [1] = {}, [2] = {}".format(idx[0],idx[1],idx[2]))
                        print("   sampIdx[0] = {}, [1] = {}, [2] = {}".format(sampIdx[idx[0]],sampIdx[idx[1]],sampIdx[idx[2]]))
                        print("   rot0[0] = {}, 1 = {}, 2 = {}".format(rot0,rot1,rot2))


def printDatasetInfo(data_set):
    
    nSamples = data_set.numSamples
    nBatches = nSamples / data_set.batchSize
    batchSizeBytes = data_set.dataSizeNBytes   # sum([x.nbytes for x in data_set.x]) / nBatches
    print("  nTrainSamples: {}".format(nSamples))
    print("  nTrainBatches: {}".format(nBatches))
    print("  batchSize: {}".format(data_set.batchSize))
    print("  batchSizeMBytes: {}mb".format(batchSizeBytes/(1024.*1024.)))
    print("  input data shape: {}".format(data_set.xShape))
                
    
    
def makeTrainsetSizeAMultipleOfValsetSize(train_set,val_set):
    nTrainSamples = train_set.numSamples
    nTrainBatches = nTrainSamples / train_set.batchSize
    #batchSizeBytes = train_set.x.nbytes / nTrainBatches

    nValSamples = val_set.numSamples
    nValBatches = nValSamples / val_set.batchSize

    print("--- valset   nsamp {},  batchsize {}".format(nValSamples,val_set.batchSize))
    print("--- trainset nsamp {},  batchsize {}".format(nTrainSamples,train_set.batchSize))
    if numpy.mod(nTrainBatches,nValBatches) != 0:
        if nTrainBatches > nValBatches:
            n = numpy.ceil( nTrainBatches / float(nValBatches) )
            d1 = numpy.floor(nValBatches*n - nTrainBatches).astype(int)   # d1 = number of batches to add to train
            d2 = 0
            while numpy.mod(nTrainBatches,nValBatches+d2) > 0:  # d2 = number of add to val to make val size a divisor of train size
                d2 += 1
            print("d1 {}, d2 {}".format(d1,d2))
            if d1 < d2:
                # add d1 random minibatches to the end of train
                print("--> stacking up train_set by {}, to {} (val {})".format(d1,nTrainBatches+d1,nValBatches))
                idx = numpy.random.permutation(nTrainBatches)
                idx = idx[:d1]
                train_set.extendSetBySubset(idx)
            else:
                # add d2 random minibatches to the end of val
                print("--> stacking up val_set by {}, to {} (train {})".format(d2,nValBatches+d2,nTrainBatches))
                idx = numpy.random.permutation(nValBatches)
                idx = idx[:d2]
                val_set.extendSetBySubset(idx)
        else:
            n = numpy.ceil( nValBatches / float(nTrainBatches) )
            d1 = numpy.floor(nTrainBatches*n - nValBatches).astype(int)   # d1 = number of batches to add to val
            d2 = 0
            while numpy.mod(nValBatches,nTrainBatches+d2) > 0:  # d2 = number of add to train to make val size a divisor of val size
                d2 += 1
            print("reverse, d1 {}, d2 {}".format(d1,d2))
            if d1 < d2:
                # add d1 random minibatches to the end of val
                print("--> stacking up val_set by {}, to {} (train {})".format(d1,nValBatches+d1,nTrainBatches))
                idx = numpy.random.permutation(nValBatches)
                idx = idx[:d1]
                val_set.extendSetBySubset(idx)
            else:
                # add d2 random minibatches to the end of train
                print("--> stacking up train_set by {}, to {} (val {})".format(d2,nTrainBatches+d2,nValBatches))
                idx = numpy.random.permutation(nTrainBatches)
                idx = idx[:d2]
                train_set.extendSetBySubset(idx)
                
        nTrainSamples = train_set.numSamples
        nTrainBatches = nTrainSamples / train_set.batchSize

        nValSamples = val_set.numSamples
        nValBatches = nValSamples / val_set.batchSize
        print("--- valset   nsamp {},  batchsize {}".format(nValSamples,val_set.batchSize))
        print("--- trainset nsamp {},  batchsize {}".format(nTrainSamples,train_set.batchSize))

    return train_set, val_set

    #macroBatchSize = nTrainBatches


def prepareData(datasetPklFileName):

    if not os.path.isfile(datasetPklFileName):
        raise ValueError("File does not exist: '{}'".format(datasetPklFileName))
        
    print ('Loading data_set from pkl {}'.format(datasetPklFileName) )

    data_set = BatchedDataset.loadFromPkl(datasetPklFileName)
    
    print("---------------------------------------")
    print("Dataset info:")
    printDatasetInfo(data_set)
    print("")
    
    return data_set
        
    
def setupLogging():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
if __name__ == '__main__':

    setupLogging()
    
    if len(sys.argv) > 1:
        trainDataFile = sys.argv[1]
    else:
        print "Error: need train data file as argument"
        sys.exit(1)
    
    v = sys.version_info
    print('running on python {}.{}.{}'.format(v.major,v.minor,v.micro))
    
    #rng = numpy.random.RandomState(23455)
    rng = numpy.random.RandomState(12345)
    #theano.config.compute_test_value = 'warn'
    theano.config.exception_verbosity = 'high'
    
    startTimePidStr = nowPidStr()
    
    #
    start_time = time.clock()
    
    #############################################################################
    #  PREPARE DATA  
    train_set = prepareData(trainDataFile)

    #############################################################################
    #  VISUALIZATIONS  
    
    #montageLinemodTrainSetSample(train_set)
    #cv2.waitKey()
    
    montageLinemodTrainSamplesAndTmpls(train_set,"tmp_out.png")
    cv2.waitKey()
    #raise ValueError("stop here")
    
    #visBatchedTrainDatasetCamPoses(train_set)
    #plt.show()
   
    ##################
    end_time = time.clock()
    print ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs'% ((end_time - start_time)) )        
    print("done")
    

    ch = 0
    while ((ch & 0xFF) != ord('q')) and (ch >= 0):
        print("waiting for 'q'")
        ch = cv2.waitKey() 
        print(" ... got '{}' ({})".format(chr(ch & 0xFF),ch))
    
    cv2.destroyAllWindows()
    
