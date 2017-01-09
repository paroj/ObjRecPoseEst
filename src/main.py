'''
Created on Oct 21, 2014

@author: wohlhart

    Train on some of testdata (real kinect sequences) + some synth train + synthetic templates 
    
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # @UnresolvedImport @UnusedImport

import sys, os, time, numpy, cv2, cPickle, copy 
import theano
import scipy.spatial.distance
import ConfigParser

from tnetcore.network import NetworkParams, Network
from tnetcore.trainer import NetworkTrainingParams, NetworkTrainer

import data
from data.basetypes import NamedImgSequence
from data.importers import LinemodTrainDataImporter
from data.batcheddatasets import dataArrayFromPatch, BatchedLinemodTrainDataset

from util.vis import calcAndVis3DDescriptorsWithRotWithTrainer, visNetworkFiltersMontage
from util.misc import nowPidStr, addNoiseToTrainSeqs
from tnetcore.networks.simpleff import SimpleFFNetwork, SimpleFFNetworkParams
from tnetcore.util import readCfgParam, readCfgIntParam, readCfgFloatParam

    
def loadData(lmDataBasepath=None,targetSize=64,inputMode=2, trainSplitFactor = 1/2., fixedNumTrain = 0, seqNames=[],zRotInv=[]):
    '''
    trainSplitFactor ... take half of the data for training,
    fixedNumTrain    ... or take (set zero to activate trainSplitFactor)
    '''
    
    if not seqNames:
        seqNames = ("ape","benchviseblue","bowl","cam","can","cat","cup","driller","duck","eggbox","glue","holepuncher","iron","lamp","phone")
        #zRotInv = (False,False,True,False,False,False,False,False,False,False,False,False,False,False,False) 
        zRotInv = (0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0)
        #seqNames = ("ape","bowl","cat","duck")
        #zRotInv = (False,True,False,False) 

    imgNumsTemplates = numpy.arange(301)
    imgNumsTrain = numpy.arange(1241)+301
    
    # DEBUG, small dataset
    #imgNumsTemplates = numpy.arange(3)
    #imgNumsTrain = numpy.arange(5)+301
    
    #targetSize = 128

    print("lmDataBasepath {}".format(lmDataBasepath))
    
    lineModTrainImgporter = LinemodTrainDataImporter(lmDataBasepath)
    
    # TEMPLATES
    tmplSeqs = dict()
    for i in range(len(seqNames)):
        seqN = seqNames[i]
        zri = zRotInv[i]
        tmplSeqs[seqN] = lineModTrainImgporter.loadSequence(objName=seqN,imgNums=imgNumsTemplates,cropSize=20.0,targetSize=targetSize,zRotInv=zri)
        
        #DEBUG
        #rots = numpy.array([patch.frame.pose.relCamPosZRotInv for patch in tmplSeqs[seqN].data])
        #print("rots.shape {}".format(rots.shape))
        #print("rots mn/mx x {},{}, y {},{}, z:{},{}".format(numpy.min(rots[:,0]),numpy.max(rots[:,0]),numpy.min(rots[:,1]),numpy.max(rots[:,1]),numpy.min(rots[:,2]),numpy.max(rots[:,2])))
    
    # TRAIN data
    trainSeqs = dict()
    for i in range(len(seqNames)):
        seqN = seqNames[i]
        zri = zRotInv[i]
        trainSeqs[seqN] = lineModTrainImgporter.loadSequence(objName=seqN,imgNums=imgNumsTrain,cropSize=20.0,targetSize=targetSize,zRotInv=zri)

    # TRAIN data, TEST data
    #traindata_set = LinemodBatchedTestData()
    #testdata_set = LinemodBatchedTestData()
    
    #trainSeqs = dict()
    testSeqs = dict()
    
    lmi = data.importers.LinemodImporter(lmDataBasepath)
    for i in range(len(seqNames)):
        seqName = seqNames[i]
        seq = lmi.loadSequence(seqName, zRotInv=zRotInv[i], inputMode=inputMode, cropAtGtPos=True,cropSize=20.,targetSize=targetSize)
        
        #cv2.imshow("test_{}".format(seqName),seq.data[0].dpt/2+0.5)
        #cv2.waitKey()
        
        # distr to trainSeqs and testSeqs
        numFrames = len(seq.data)
        print("{}: {}".format(i,numFrames))
        
        # select train and test idx:
        # chose a certain number of train for each tmpl position
        
        # step 0, get poses of tmpls
        tmplSeq = tmplSeqs[seqName]
        numTmpl = len(tmplSeq.data)
        tmplRots = numpy.array([patch.frame.pose.relCamPosZRotInv for patch in tmplSeq.data])
        tmplRots = tmplRots / numpy.tile(numpy.sqrt((tmplRots**2).sum(axis=1)),(3,1)).T
        
        # step 1, find closest templates
        rots = numpy.array([patch.frame.pose.relCamPosZRotInv for patch in seq.data])
        rots = rots / numpy.tile(numpy.sqrt((rots**2).sum(axis=1)),(3,1)).T
        
        #DEBUG
        print("rots mn/mx x {},{}, y {},{}, z:{},{}".format(numpy.min(rots[:,0]),numpy.max(rots[:,0]),numpy.min(rots[:,1]),numpy.max(rots[:,1]),numpy.min(rots[:,2]),numpy.max(rots[:,2])))
        #assert not zRotInv[i], "stop here"
            
        
        sim = numpy.zeros((numTmpl,numFrames),dtype=numpy.float)
        for j in xrange(numTmpl):
            tmplRot = tmplRots[j]
            sim[j,:] = numpy.dot(rots,tmplRot) 
            if zRotInv[i] == 2:   # half rot invariant
                sim[j,:] = numpy.maximum(sim[j,:], numpy.dot(rots*numpy.array([-1,-1,1]),tmplRot) )

        # for every image the idx of the closest tmpl 
        maxSimTmplIdx = numpy.argmax(sim, axis=0)    
        #print("maxSimTmplIdx {}:".format(maxSimTmplIdx))            
        
        # step 2, take training samples for each template, make the rest test
        #numTrainPerTmpl = (numpy.tile(numpy.arange(numTmpl).reshape((numTmpl,1)),(1,numFrames)) == numpy.tile(maxSimTmplIdx.reshape((1,numFrames)),(numTmpl,1))).sum(axis=1)
        #print("numTrainPerTmpl mn/mx {},{}".format(numpy.min(numTrainPerTmpl),numpy.max(numTrainPerTmpl)))
        #print("{}".format(numTrainPerTmpl))
        
        trainIdx = numpy.zeros(numFrames,dtype=numpy.int)
        testIdx = numpy.zeros(numFrames,dtype=numpy.int)
        k = 0
        l = 0
        for j in xrange(numTmpl):
            # for every tmpl add 1/2 of the samples as train, rest to test
            frIdx, = (maxSimTmplIdx == j).nonzero()  # frames that have j as most similar template
            #print("frIdx {}, frIdx {}".format(len(frIdx),frIdx))
            frIdx = numpy.random.permutation(frIdx)
            if len(frIdx) > 0:
                if fixedNumTrain > 0:
                    nTrn = numpy.minimum(len(frIdx),fixedNumTrain)
                else:
                    nTrn = numpy.ceil(len(frIdx)*trainSplitFactor).astype(numpy.int)
                nTst = len(frIdx) - nTrn 
                #print("nTrn {}, nTst {}, frIdx {}, frIdx {}".format(nTrn,nTst,len(frIdx),frIdx))
                #print("{}, {}, {}, {}".format(trainIdx.shape,trainIdx[k:k+nTrn].shape,frIdx.shape,frIdx[0:nTrn].shape))
                trainIdx[k:k+nTrn] = frIdx[0:nTrn]
                k += nTrn
                if nTst > 0:
                    testIdx[l:l+nTst] = frIdx[nTrn:nTrn+nTst]
                    l += nTst
        trainIdx = trainIdx[0:k]
        testIdx = testIdx[0:l]
        
        print("class {}, nTrain {}, nTest {}".format(i,k,l))
            
        #print("{}".format(trainIdx))
        #print("{}".format(testIdx))
        
        trainSeqData = [seq.data[j] for j in trainIdx]
        trainSeqs[seqName].data.extend(trainSeqData)         
        
        testSeqs[seqName] = NamedImgSequence(seqName,[seq.data[j] for j in testIdx])
        
    #raise ValueError("stop here")

    # DEBUG show input data
    #montageLinemodSamples(tmplSeqs[seqNames[0]])
    #cv2.waitKey()

    return tmplSeqs,trainSeqs,testSeqs


def zeroSaveRandInt(n):
    if n >= 1:
        return numpy.random.randint(n)
    else:
        return 0

def createTripleAndPairsDatasetFromImgSeq(tmplSeqs,trainSeqs,inputMode=0,batchSize=None):
    
    print("*********************************************************************")
    print("          start createTripleDatasetFromImgSeq")
    print("*********************************************************************")
    
    floatX = theano.config.floatX  # @UndefinedVariable
    
    #numSeqs = len(tmplSeqs)
    seqNames = tmplSeqs.keys() 
    seqNames = sorted(seqNames)
    numClasses = len(seqNames)
    
    # construct initial set of triplets:
    #   loop through classes
    
    #for seqN in trainSeqs:
    #    print("{}: {}".format(seqN,len(trainSeqs[seqN].data)))
            
    numTrainPerClass = numpy.array([len(trainSeqs[seqN].data) for seqN in trainSeqs])
    assert numpy.count_nonzero(numTrainPerClass - numTrainPerClass[0]) == 0, "cannot deal with unequal training sequence sizes"
    numTrainPerClass = numpy.max(numTrainPerClass)
    
    numTmplPerClass = len(tmplSeqs[seqNames[0]].data)
    #numTrainPerClass = len(trainSeqs[seqNames[0]].data)
    print "num train ", numTrainPerClass
    print "num tmpl ", numTmplPerClass
    
    # calculate similarity of poses of train sample and templates of the sample class 
    seqName0 = seqNames[0]
    tmplSeq0 = tmplSeqs[seqName0]
    #trainSeq0 = trainSeqs[seqName0]
    sim = numpy.zeros((numClasses,numTmplPerClass,numTrainPerClass),dtype=numpy.float)
    
    #tmplRots = numpy.array([patch.frame.pose.relCamPosZRotInv for patch in tmplSeq0.data])
    #tmplRots = tmplRots / numpy.tile(numpy.sqrt((tmplRots**2).sum(axis=1)),(3,1)).T
    
    zRotInv = []
    for c in xrange(numClasses):
        zRotInv.append(tmplSeqs[seqNames[c]].data[0].frame.pose.zRotInv)
    print("zRotInv: {}".format(zRotInv))
    #raise ValueError("stop here")
    
    tmplRots = numpy.zeros((numClasses,numTmplPerClass,3),dtype=numpy.float)
    #tmplZRI = numpy.zeros((numClasses,numTmplPerClass,),dtype=numpy.uint8)
    trainRots = numpy.zeros((numClasses,numTrainPerClass,3),dtype=numpy.float)
    #trainZRI = numpy.zeros((numClasses,numTrainPerClass,),dtype=numpy.uint8)
    for c in xrange(numClasses):
        
        trots = numpy.array([patch.frame.pose.relCamPosZRotInv for patch in tmplSeqs[seqNames[c]].data])
        tmplRots[c] = trots / numpy.tile(numpy.sqrt((trots**2).sum(axis=1)),(3,1)).T  
        trots = numpy.array([patch.frame.pose.relCamPosZRotInv for patch in trainSeqs[seqNames[c]].data])
        trainRots[c] = trots / numpy.tile(numpy.sqrt((trots**2).sum(axis=1)),(3,1)).T
        for i in xrange(numTmplPerClass):
            sim[c,i,:] = numpy.dot(trainRots[c],tmplRots[c,i]) 
            if zRotInv[c] == 2:
                sim[c,i,:] = numpy.maximum(sim[c,i,:], numpy.dot(trainRots[c]*numpy.array([-1,-1,1]),tmplRots[c,i]))
        
    # for every training image idx of the closest tmpl 
    maxSimTmplIdx = numpy.argmax(sim, axis=1)
    
    #print ("maxSimTmplIdx.shape {}".format(maxSimTmplIdx.shape))
    #raise ValueError("stop here")
        
    # DEBUG, show random 200 train img and its closest tmpl. 
    #   imgs:128x128, img+tmpl=256x128 ->  in area 1280x1024 -> 10 cols, 4 rows 
#     debugImg = numpy.zeros((1024,1280),dtype=numpy.float)
#     idx = numpy.random.permutation(numTrainPerClass)
#     nc = 2
#     tmplSeqC = tmplSeqs[seqNames[nc]]
#     trainSeqC = trainSeqs[seqNames[nc]]
#     print("trainSeqC.data[0] {}".format(type(trainSeqC.data[0])))
#     print("trainSeqC.data[0].img {}".format(type(trainSeqC.data[0].img)))
#     print("trainSeqC.data[0].dpt {}".format(type(trainSeqC.data[0].dpt)))
#     s = trainSeqC.data[0].dpt.shape[0]
#     numCols = 1280/s
#     numRows = 1024/(s*2)  
#     print "s ", s, numCols, numRows
#     for x in xrange(numCols):
#         for y in xrange(numRows):
#             n = idx[y*10+x]
#             img = trainSeqC.data[n].dpt 
#             tmplImg = tmplSeqC.data[maxSimTmplIdx[nc,n]].dpt 
#             debugImg[y*s*2:(y+1)*s*2,x*s:(x+1)*s] = numpy.concatenate((img,tmplImg),axis=0)/2. + 0.5
#             #print "sim ", n, sim[maxSimTmplIdx[n],n]
#     cv2.imshow("tmpls",debugImg)
#     cv2.waitKey()
    
    ##########
    print("create randIdx")
    #n = 0
    #numTmpls = numpy.zeros(numClasses) # how many tmpls were chosen in this batch
    randIdx = numpy.zeros((numClasses,numTrainPerClass),dtype=numpy.int)
    for i in xrange(numClasses):
        # init random permutations of the training samples
        randIdx[i,:] = numpy.random.permutation(numTrainPerClass)
        #print("{}, {}".format(i,randIdx[i]))

    #
    if inputMode == 0:          
        imgSize = numpy.array(tmplSeq0.data[0].dpt.shape)
    elif inputMode == 1:
        imgSize = numpy.array(tmplSeq0.data[0].img.shape)
    else:
        imgSize = numpy.array(tmplSeq0.data[0].img.shape)
        imgSize[2] += 1 # add depth
        
    #print("imgSize {}".format(imgSize))
    #print("imgSize {}".format(type(imgSize)))
    #print("imgSize.size {}".format(imgSize.size))
    if imgSize.size > 2:
        nChan = imgSize[2]  # IMG, DPT, IMG + DPT ?
    else:
        nChan = 1
    
    #print("nChan {}".format(nChan))    
    
    if batchSize is None:
        batchSize = 300 #256 #1024
    #numBatches = ?
    nTripletsPerBatch = batchSize*3 # TODO ?
    nPairsPerBatch = batchSize
     
    batchData = numpy.zeros((batchSize,nChan,imgSize[0],imgSize[1]),dtype=floatX)
    batchLabels = numpy.zeros((batchSize,),dtype=numpy.int32)
    batchTripletIdx = numpy.zeros((nTripletsPerBatch,3),dtype=numpy.int32)
    batchPairIdx = numpy.zeros((nPairsPerBatch,2),dtype=numpy.int32)
    batchMaxSimTmplIdx = numpy.zeros((batchSize,),dtype=numpy.int32)
    batchSampIdx = numpy.zeros((batchSize,),dtype=numpy.int32)  # index of the training samples and tmpls that landed in the batch
    batchRots = numpy.zeros((batchSize,3),dtype=numpy.float32)
         
    maxNumBatches = numpy.ceil((3*numTrainPerClass*numClasses)/float(batchSize))  # maximally every trainSample and its closest tmpl + one random -> 3*numSamples/batchSize
    allData = numpy.zeros((maxNumBatches*batchSize,nChan,imgSize[0],imgSize[1]),dtype=floatX)
    allLabels = numpy.zeros((maxNumBatches*batchSize,),dtype=numpy.int32)
    allTripletIdx = numpy.zeros((maxNumBatches*nTripletsPerBatch,3),dtype=numpy.int32)
    allPairIdx = numpy.zeros((maxNumBatches*nPairsPerBatch,2),dtype=numpy.int32)
    allMaxSimTmplIdx = numpy.zeros((maxNumBatches*batchSize,),dtype=numpy.int32)
    allSampIdx = numpy.zeros((maxNumBatches*batchSize,),dtype=numpy.int32)    # index of the training samples and tmpls that landed in the batch
    allTmplBatchDataStartIdx = numpy.zeros((maxNumBatches,numClasses),dtype=numpy.int32)
    allRots = numpy.zeros((maxNumBatches*batchSize,3),dtype=numpy.float32)
    
    
    mb = allData.nbytes/(1024.*1024.)
    print("max allData size: {}Mb".format(mb))
       
    numBatches = 0
    trainSampIdx = 0 
    currentClass = 0
    allTrainSamplesUsed = False
    while not allTrainSamplesUsed:  # keep creating batches until no more data  #  (trainSampIdx < numTrainPerClass) 
        
        #print("############### START BATCH #################################")
        totalNumChosenTmpl = 0
        chosenTmpls = []
        for c in xrange(numClasses):
            chosenTmpls.append(set([])) # init empty set of chosen tmpls
        
        k = 0 # idx within batch where the next sample (train and further down template) goes        
        while k+totalNumChosenTmpl < batchSize:
            # from each class
            
            #for c in xrange(numClasses):
            c = currentClass
            
            seqN = seqNames[c]
            #tmplSeq = tmplSeqs[seqN]
            trainSeq = trainSeqs[seqN]
            
            # take one training image (next one according to random index)
            sampIdx = randIdx[c,trainSampIdx]
            trainImg = trainSeq.data[sampIdx]
            
            # take its template
            tmplIdx = maxSimTmplIdx[c,sampIdx]
            # add template to set of chosen templates
            # and update number of chosen templates
            totalNumChosenTmpl -= len(chosenTmpls[c])
            chosenTmpls[c] = chosenTmpls[c] | {tmplIdx} 
            totalNumChosenTmpl += len(chosenTmpls[c])
            
            imgD = dataArrayFromPatch(trainImg,inputMode)
                
            batchData[k] = imgD
            batchMaxSimTmplIdx[k] = tmplIdx  
            batchLabels[k] = c 
            batchSampIdx[k] = sampIdx
            batchRots[k] = trainRots[c,sampIdx]
            
            k += 1
            
            #print("currentClass {}, sampIdx {}, trainSampIdx {}, tmplIdx {}".format(currentClass,sampIdx,trainSampIdx,tmplIdx))

            currentClass += 1
            if currentClass >= numClasses:
                currentClass = 0
                trainSampIdx += 1
                if trainSampIdx >= numTrainPerClass:
                    print("no more training samples -> start from beginning, but set a flag so we know this is the last batch")
                    allTrainSamplesUsed = True
                    trainSampIdx = 0
                    #break
            
            if k+numpy.maximum(totalNumChosenTmpl,2*numClasses) >= batchSize:  # we need space for the chosen, or at least two templates of each class
                # then we're done
                break

#         print("----- totalNumChosenTmpl: {}".format(totalNumChosenTmpl))            
#         for cx in xrange(numClasses):
#             print("{}: {}".format(cx,len(chosenTmpls[cx])))
#         print("---------")
        
        batchNumTrainSamples = k
        #print("batchNumTrainSamples {}".format(batchNumTrainSamples))
        #print("Templates:")
        #for i in range(len(chosenTmpls)): 
        #    print("{}, {}: {}".format(i,len(chosenTmpls[i]),chosenTmpls[i])) 
            
        for c in range(len(chosenTmpls)):
            ct = chosenTmpls[c]
            chosenTmpls[c] = numpy.array(list(ct)).astype(numpy.int32)
            if len(chosenTmpls[c]) < 2:
                totalNumChosenTmpl -= len(chosenTmpls[c])
                # include two random templates for each class
                if len(chosenTmpls[c]) == 0:
                    chosenTmpls[c] = numpy.random.choice(numTmplPerClass,2,replace=False)
                else:
                    x = numpy.random.randint(numTmplPerClass-1)
                    if x >= chosenTmpls[c]:
                        x += 1
                    chosenTmpls[c] = numpy.concatenate((chosenTmpls[c],numpy.array([x])))
                totalNumChosenTmpl += len(chosenTmpls[c])
                    
            numpy.random.shuffle(chosenTmpls[c])
        
        #print("Templates after stacking up to minimum 2 tmpls per class:")
        #for i in range(len(chosenTmpls)): 
        #    print("{}, {}: {} ({})".format(i,len(chosenTmpls[i]),chosenTmpls[i],type(chosenTmpls[i]))) 
        
        #########################################################################
        # add chosen templates to the batch data
        #    - up to k, everything is filled up with train samples
        
        #    - first check if we can fit all chosen templates
        while batchNumTrainSamples + totalNumChosenTmpl > batchSize:
            # cut away chosen templates. start with the class that has most templates
            maxCTmpls = 0
            maxCTmplsIdx = 0
            for i in range(len(chosenTmpls)):
                ct = chosenTmpls[i]
                if len(ct) > maxCTmpls:
                    maxCTmpls = len(ct)
                    maxCTmplsIdx = i
            #print("maxCTmpls {}".format(maxCTmpls))
            #print("maxCTmplsIdx {}".format(maxCTmplsIdx))
            #print("chosenTmpls[maxCTmplsIdx] {}".format(chosenTmpls[maxCTmplsIdx]))
            #print("t chosenTmpls[maxCTmplsIdx] {}".format(type(chosenTmpls[maxCTmplsIdx])))
            chosenTmpls[maxCTmplsIdx] = chosenTmpls[maxCTmplsIdx][:-1]
            totalNumChosenTmpl -= 1
            
        for i in range(len(chosenTmpls)): 
            chosenTmpls[i] = numpy.sort(chosenTmpls[i]) 

#         print("Templates after cutting to max batchSize:")
#         for i in range(len(chosenTmpls)): 
#             print("{}, {}: {} ({})".format(i,len(chosenTmpls[i]),chosenTmpls[i],type(chosenTmpls[i]))) 
                            
                            
        # check that for each class of which we have a label in the batch there is at least one template
        for i in range(batchNumTrainSamples):
            if len(chosenTmpls[batchLabels[i]]) <= 0:
                raise ValueError("no template for class {}".format(batchLabels[i]))
            if len(chosenTmpls[batchLabels[i]]) < 2:
                raise ValueError("only one template for class {}".format(batchLabels[i]))
                                
        #print("add templates to batch")                        
        tmplBatchDataStartIdx = numpy.zeros(numClasses,dtype=numpy.int32)
        tmplBatchDataIdx = -numpy.ones((numClasses,numTmplPerClass),dtype=numpy.int32) # for every chosen tmpl the index where it ends up in the batch (-1 for all others)
        for c in xrange(numClasses):

            #print("Add templates for class {}: {}".format(c,chosenTmpls[c]))
            #print("  current k={}".format(k))
            
            #assert len(chosenTmpls[c]) > 0
            if len(chosenTmpls[c]) > 0:
            
                tmplBatchDataStartIdx[c] = k
                tmplSeq = tmplSeqs[seqNames[c]]
                chosenTmpls[c] = numpy.array(list(chosenTmpls[c])).astype(numpy.int32)
                tmplBatchDataIdx[c,chosenTmpls[c]] = numpy.arange(len(chosenTmpls[c]))+k 
                for l in xrange(len(chosenTmpls[c])):

                    if k >= batchSize:
                        for cx in xrange(numClasses):
                            print("{}: {}".format(cx,len(chosenTmpls[cx])))
                        raise ValueError("Cannot fit all chosen tmpls in batch. This shouldn't happen anymore ({},{},{},{},{},{})".format(k,batchSize,l,c,batchNumTrainSamples,totalNumChosenTmpl))

                    cTmplIdx = chosenTmpls[c][l]
                    
                    imgD = dataArrayFromPatch(tmplSeq.data[cTmplIdx],inputMode)
                        
                    batchData[k] = imgD 
                    batchLabels[k] = c
                    batchSampIdx[k] = cTmplIdx
                    batchRots[k] = tmplRots[c,cTmplIdx]
                    k += 1
            
            
        #print("tmplBatchDataIdx {}".format(tmplBatchDataIdx))
        
        ##################################################################################        
        #
        #   CREATE TRIPLET INDEX for this batch
        #
        ##################################################################################
        
        # create three types of triplet indizes
        #  1.) samp, maxSimTmpl, secondMostSim (or random for now)
        #  2.) samp, maxSimTmpl, randomSameClassTmpl
        #  3.) samp, maxSimTmpl, randomOtherClassTmpl
        
        # first is always the sample
        batchTripletIdx[0:batchNumTrainSamples*3,0] = numpy.tile(numpy.arange(batchNumTrainSamples),(1,3))
        
        # second is the maxSimTmpl
        #    it was batchMaxSimTmplIdx, but where did it end up in the batchData?
        maxSimTmplDataIdx = numpy.zeros((batchNumTrainSamples,),dtype=numpy.int32) 
        for i in xrange(batchNumTrainSamples):
            c = batchLabels[i]
            maxSimTmplDataIdx[i] = tmplBatchDataIdx[c][batchMaxSimTmplIdx[i]]
            if maxSimTmplDataIdx[i] < 0:  # we couldn't fit its closest into the batch, then what? -> take the best of those that are there
                availableSameClassTmpls = chosenTmpls[c]
                #print("max sim tmpl {} (class {}) not in batch; these are available: {}".format(batchMaxSimTmplIdx[i],c,chosenTmpls[c]))
                maxSimTmplDataIdx[i] = numpy.argmax(sim[c,availableSameClassTmpls,batchSampIdx[i]]) + tmplBatchDataStartIdx[c]
                #print(" -> choose {}".format(maxSimTmplDataIdx[i]))
                
        batchTripletIdx[0:batchNumTrainSamples*3,1] = numpy.tile(maxSimTmplDataIdx,(1,3))
        
        # 1.) third is the second most sim (?) ... or also random for now; 
        #         also (mainly for zRotInv): dont take one with exactly the same rot.
        for i in xrange(batchNumTrainSamples):
            c = batchLabels[i]
            # eligible other tmpls from same class: 
            otherIdx = numpy.arange(len(chosenTmpls[c])) + tmplBatchDataStartIdx[c] # the chosen ones from the same class
            # remove those that have the same similarity to train as the closest tmpl
            trIdx = batchSampIdx[batchTripletIdx[i,0]]
            tmpl1Idx = batchSampIdx[batchTripletIdx[i,1]]
            otherTmplIdx = batchSampIdx[otherIdx]
            t1Sim = sim[c,tmpl1Idx,trIdx]
            otherTmplSim = sim[c,otherTmplIdx,trIdx] 
            otherIdx = otherIdx[otherTmplSim < t1Sim]
            if len(otherIdx) > 0:
                batchTripletIdx[i,2] = numpy.random.choice(otherIdx)
            else:  
                # we dont have any other from the same class with different pose in this batch. 
                # that's not good, but for now lets just take one of another class
                otherClass = zeroSaveRandInt(numClasses-1)
                if otherClass >= c:
                    otherClass += 1 
                batchTripletIdx[i,2] = zeroSaveRandInt(len(chosenTmpls[otherClass])) + tmplBatchDataStartIdx[otherClass]
                
            
        # 2.) third is a random tmpl from the same class (but preferably not the maxSim one)
        #                                   #         also (mainly for zRotInv): dont take one with exactly the same rot.
        for i in xrange(batchNumTrainSamples):
            c = batchLabels[i]
            tidx = batchNumTrainSamples + i
            # eligible other tmpls from same class: 
            otherIdx = numpy.arange(len(chosenTmpls[c])) + tmplBatchDataStartIdx[c] # the chosen ones from the same class
            # remove those that have the same similarity to train as the closest tmpl
            trIdx = batchSampIdx[batchTripletIdx[tidx,0]]
            tmpl1Idx = batchSampIdx[batchTripletIdx[tidx,1]]
            otherTmplIdx = batchSampIdx[otherIdx]
            t1Sim = sim[c,tmpl1Idx,trIdx]
            otherTmplSim = sim[c,otherTmplIdx,trIdx] 
            otherIdx = otherIdx[otherTmplSim < t1Sim]
            if len(otherIdx) > 0:
                batchTripletIdx[tidx,2] = numpy.random.choice(otherIdx) 
            else:  
                # we dont have any other from the same class with different pose in this batch. 
                # that's not good, but for now lets just take one of another class
                otherClass = zeroSaveRandInt(numClasses-1)
                if otherClass >= c:
                    otherClass += 1 
                batchTripletIdx[tidx,2] = zeroSaveRandInt(len(chosenTmpls[otherClass])) + tmplBatchDataStartIdx[otherClass]
        
        # 3.) third is a random tmpl from another class
        for i in xrange(batchNumTrainSamples):
            c = batchLabels[i]
            tidx = i + 2*batchNumTrainSamples
            otherClass = zeroSaveRandInt(numClasses-1)
            if otherClass >= c:
                otherClass += 1 
            batchTripletIdx[tidx,2] = zeroSaveRandInt(len(chosenTmpls[otherClass])) + tmplBatchDataStartIdx[otherClass]
        
        # couple more (3*(batchSize-batchNumTrainSamples)) triplets allowed, 
        #  so what to do with them ? -> bootstrap 
        #   ... or more randoms for now: lets do the first is a random from the same class, the second is completely random
        for i in xrange(batchNumTrainSamples*3,nTripletsPerBatch):
            sIdx = i % batchNumTrainSamples
            c = batchLabels[sIdx] 
            trIdx = batchSampIdx[sIdx]
            batchTripletIdx[i,0] = sIdx
            batchTripletIdx[i,1] = zeroSaveRandInt(len(chosenTmpls[c])) + tmplBatchDataStartIdx[c]  # random tmpl of same class 
            c2 = zeroSaveRandInt(numClasses)
            if c2 == c:
                batchTripletIdx[i,2] = zeroSaveRandInt(len(chosenTmpls[c])-1) + tmplBatchDataStartIdx[c] 
                if (batchTripletIdx[i,2] >= batchTripletIdx[i,1]) and (len(chosenTmpls[c]) > 1):
                    batchTripletIdx[i,2] += 1
                # now check if the tmpl in batchTripletIdx[i,2] is actually more similar than batchTripletIdx[i,1]
                if (batchTripletIdx[i,1] < 0) or (batchTripletIdx[i,1] >= len(batchLabels)):
                    print("------- FAIL, triplet idx[1] wrong")
                    print("{}, {}".format(batchTripletIdx[i,1],len(batchLabels)))
                    print(batchTripletIdx)
                    print(batchLabels)
                    print(chosenTmpls)
                    print(tmplBatchDataStartIdx)
                    assert False
                assert batchLabels[batchTripletIdx[i,1]] == c, "batchlabels {}, batchTripletIdx {}".format(batchLabels,batchTripletIdx)
                #if batchLabels[batchTripletIdx[i,2]] != c2:
                #    print("batchTripletIdx[i,2] = {}".format(batchTripletIdx[i,2]))
                #    print("tmplBatchDataStartIdx[c] {}".format(tmplBatchDataStartIdx[c]))
                #    print("batchLabels[tmplBatchDataStartIdx[c]] {}, c {}".format(batchLabels[tmplBatchDataStartIdx[c]],c))
                assert batchLabels[batchTripletIdx[i,2]] == c2, "{}, {}, {}, {}".format(batchLabels[batchTripletIdx[i,2]],c2,c,len(chosenTmpls[c]))
                s1 = batchSampIdx[batchTripletIdx[i,1]]
                s2 = batchSampIdx[batchTripletIdx[i,2]]
                if sim[c,s2,trIdx] > sim[c,s1,trIdx]: # then we have to swap them
                    h = batchTripletIdx[i,2]
                    batchTripletIdx[i,2] = batchTripletIdx[i,1]
                    batchTripletIdx[i,1] = h
                #else:
                #    if sim[c,s2,trIdx] == sim[c,s1,trIdx]: # then we have to find another one
            else:
                batchTripletIdx[i,2] = zeroSaveRandInt(len(chosenTmpls[c2])) + tmplBatchDataStartIdx[c2]
        
        ##################################################################################        
        #
        #   CREATE PAIR INDEX for this batch
        #
        ##################################################################################

        # essentially link each train example to its closest template
        batchPairIdx = numpy.zeros((nPairsPerBatch,2),dtype=numpy.int32)
        k = 0
        for i in range(batchNumTrainSamples):
            if maxSimTmplDataIdx[i] >= 0:
                batchPairIdx[k,0] = i
                batchPairIdx[k,1] = maxSimTmplDataIdx[i] 
                k += 1
        # TODO fill up k..nPairsPerBatch with equal (sim==1) templates 
        
        # DEBUG  print assembled batch
#         print("-----BATCH RESULTS--------------------------------------") 
#         print("sampIdx: {}".format(batchSampIdx))
#         #print("istrain: {}".format(numpy.arange(len(batchSampIdx)) < tmplBatchDataStartIdx))
#         print("labels:  {}".format(batchLabels))
#         print("triplets:")
#         print batchTripletIdx
#         print("batchMaxSimTmplIdx {}".format(batchMaxSimTmplIdx))
#         print("tmplBatchDataStartIdx {}".format(tmplBatchDataStartIdx))
#         print("-------------------------------------------") 
        
        # collect batchData into big data block
        allData[numBatches*batchSize:(numBatches+1)*batchSize] = batchData
        allLabels[numBatches*batchSize:(numBatches+1)*batchSize] = batchLabels
        allTripletIdx[numBatches*nTripletsPerBatch:(numBatches+1)*nTripletsPerBatch] = batchTripletIdx ##** + numBatches*batchSize # (absolute idx, absolutely not clever)
        allPairIdx[numBatches*nPairsPerBatch:(numBatches+1)*nPairsPerBatch] = batchPairIdx #** + numBatches*batchSize # (absolute idx, absolutely not clever)
        allMaxSimTmplIdx[numBatches*batchSize:(numBatches+1)*batchSize] = batchMaxSimTmplIdx
        allSampIdx[numBatches*batchSize:(numBatches+1)*batchSize] = batchSampIdx
        allRots[numBatches*batchSize:(numBatches+1)*batchSize] = batchRots  
        allTmplBatchDataStartIdx[numBatches] = tmplBatchDataStartIdx  #** + numBatches*batchSize  # (absolute idx, absolutely not clever)
        numBatches += 1
        
    if numBatches < maxNumBatches:
        allData = allData[:numBatches*batchSize]
        allLabels = allLabels[:numBatches*batchSize]
        allTripletIdx = allTripletIdx[:numBatches*nTripletsPerBatch]
        allPairIdx = allPairIdx[:numBatches*nPairsPerBatch]
        allMaxSimTmplIdx = allMaxSimTmplIdx[:numBatches*batchSize]
        allSampIdx = allSampIdx[:numBatches*batchSize]
        allRots = allRots[:numBatches*batchSize]  
        allTmplBatchDataStartIdx = allTmplBatchDataStartIdx[:numBatches]  
    
    # only positive/same pairs    
    allPairLabels = numpy.ones((numBatches*nPairsPerBatch,),dtype=numpy.int32)    

    #trainRots = trainRots.reshape((numTrainPerClass*numClasses,3))
    #tmplRots = tmplRots.reshape((numTmplPerClass*numClasses,3))
    sampleInfo = {'maxSimTmplIdx':allMaxSimTmplIdx, 'sampIdx':allSampIdx, 'tmplBatchDataStartIdx':allTmplBatchDataStartIdx,  'rots': allRots, 'tmplRots':tmplRots, 'trainRots':trainRots, 'nTrainPerSeq':numTrainPerClass, 'zRotInv': zRotInv }
    
    data_set = BatchedLinemodTrainDataset(allData,y=allLabels,sampleInfo=sampleInfo,pairIdx=allPairIdx,pairLabels=allPairLabels,tripletIdx=allTripletIdx,nPairsPerBatch=nPairsPerBatch,nTripletsPerBatch=nTripletsPerBatch,batchSize=batchSize)

    print("-------------------------------------------")
    print("Final dataset info: ")
    print("numSamples: {}".format(data_set.x.shape[0]))
         
    return data_set
    
    
def bootstrapTriplets(descrNetTrainer,train_set):
    '''
    Add two new triplets for each sample:
      * one against the closest descriptor of a template of another class
      * one against the closest descriptor of a template of the same class, that's not the target template
    '''
    
    print("bootstrapping")
        
    descrs = descrNetTrainer.computeDescriptors(train_set)
    
    print("train descrs computed")
    labels = train_set.y
    # for each batch get descriptors for all samples
    
    #print("descrs.shape {}".format(descrs.shape))
    #print("descrs mn/mx {},{}".format(numpy.min(descrs),numpy.max(descrs)))
    
    numSamples = train_set.x.shape[0]
    batchSize = train_set.batchSize
    numBatches = numSamples / batchSize 
    
    oldNTripletsPerBatch = train_set.nTripletsPerBatch  
    # 2 new triplets for each sample
    newNTripletsPerBatch = oldNTripletsPerBatch + 2*batchSize  
    newTripletIdx = numpy.zeros((numBatches*newNTripletsPerBatch,3),dtype=numpy.int32)
    
#     sampleInfo = {'maxSimTmplIdx':allMaxSimTmplIdx, 
#                   'sampIdx':allSampIdx, 
#                   'tmplBatchDataStartIdx':allTmplBatchDataStartIdx, 
#                   'rots': allRots, 
#                   'tmplRots':tmplRots, 
#                   'trainRots':trainRots }
        
    tmplBatchDataStartIdx = train_set.sampleInfo['tmplBatchDataStartIdx']
    zRotInv = train_set.sampleInfo['zRotInv']
    sampIdx = train_set.sampleInfo['sampIdx']
    tmplRots = train_set.sampleInfo['tmplRots']
    trainRots = train_set.sampleInfo['trainRots']
    
    for nBatch in xrange(numBatches):
        # add old triplets
        newTripletIdx[nBatch*newNTripletsPerBatch:nBatch*newNTripletsPerBatch+oldNTripletsPerBatch] = train_set.tripletIdx[nBatch*oldNTripletsPerBatch:(nBatch+1)*oldNTripletsPerBatch]

        # calc distances
        bDescrs = descrs[nBatch*batchSize:(nBatch+1)*batchSize] # prefix b = batch -> descriptors of the current batch
        bLabels = labels[nBatch*batchSize:(nBatch+1)*batchSize]
        tmplStartIdx = tmplBatchDataStartIdx[nBatch] #** - nBatch*batchSize
        #print("tmplStartIdx {} ({})".format(tmplStartIdx,type(tmplStartIdx)))
        #print("bLabels[tmplIdx] {}".format(bLabels[tmplStartIdx]))
        bNumTrain = tmplStartIdx[0]
        trainDescrs = bDescrs[:bNumTrain]
        tmplDescrs = bDescrs[bNumTrain:]
        dst = scipy.spatial.distance.cdist(trainDescrs,tmplDescrs)
        
        bTmplLabels = bLabels[bNumTrain:]
        
        #print("dst shape {}".format(dst.shape))
        
        
        # add new triplets
        bULabs = numpy.unique(bLabels)
        bSameClassTmplIdx = {}
        bOtherClassTmplIdx = {}
        for c in bULabs:
            bSameClassTmplIdx[c], = (bTmplLabels == c).nonzero()
            bOtherClassTmplIdx[c], = (bTmplLabels != c).nonzero()
            #print("c {}".format(c))
            #print("{}".format(bSameClassTmplIdx[c]))
            #print("{}".format(bOtherClassTmplIdx[c]))
    
        assert(bTmplLabels[bSameClassTmplIdx[c][0]] == c)
        assert(bTmplLabels[bOtherClassTmplIdx[c][0]] != c)
        
        # assuming there was an tripletIdx in the train_set where for each sample there was a triplet with the target tmpl as the "close" sample
        #**targetTmplIdx = train_set.tripletIdx[nBatch*oldNTripletsPerBatch:nBatch*oldNTripletsPerBatch+bNumTrain,1] - nBatch*batchSize - bNumTrain
        targetTmplIdx = train_set.tripletIdx[nBatch*oldNTripletsPerBatch:nBatch*oldNTripletsPerBatch+bNumTrain,1] - bNumTrain
            
#         k = nBatch*newNTripletsPerBatch+oldNTripletsPerBatch
#         for i in xrange(bNumTrain):  # for each training sample
#             sLabel = bLabels[i]   # prefix s = sample
#             # closest from other class
#             minDstOtherClassTmplIdx = numpy.argmin( dst[i,bOtherClassTmplIdx[sLabel]] )  
#             minDstOtherClassTmplIdx = bOtherClassTmplIdx[sLabel][ minDstOtherClassTmplIdx ]  # fuck this is strange. but it's correct  (right?) 
#             
#             newTripletIdx[k] = minDstOtherClassTmplIdx + bNumTrain + nBatch*batchSize
#             k += 1
#             
#             # set target tmpl to have a high distance, such that it wont be selected as the closest
#             dst[i,targetTmplIdx[i]] = numpy.max(dst[i,:])+1
#             minDstSameClassTmplIdx = numpy.argmin( dst[i,bSameClassTmplIdx[sLabel]] )
#             minDstSameClassTmplIdx = bSameClassTmplIdx[sLabel][ minDstSameClassTmplIdx ]
#             
#             newTripletIdx[k] = minDstSameClassTmplIdx + bNumTrain + nBatch*batchSize
#             k += 1
            
        batchTripletIdxEnd = (nBatch+1)*newNTripletsPerBatch
        k = nBatch*newNTripletsPerBatch+oldNTripletsPerBatch
        i = 0
        tmplListIdx = 0 # take closest wrong tmpl (and increment this every time we are through with one loop over all samples)
        while k < batchTripletIdxEnd:   # as long as there are empty slots for new triplets
            sLabel = bLabels[i]   # prefix s = sample .... sLabel is the sample's label
            
            #--- closest from other class
            minDstOtherClassTmplIdx = numpy.argsort( dst[i,bOtherClassTmplIdx[sLabel]] )
            #print("minDstOtherClassTmplIdx.shape {}".format(minDstOtherClassTmplIdx.shape))
            minDstOtherClassTmplIdx = minDstOtherClassTmplIdx[numpy.minimum( tmplListIdx, len(minDstOtherClassTmplIdx)-1 )]
            minDstOtherClassTmplIdx = bOtherClassTmplIdx[sLabel][ minDstOtherClassTmplIdx ]  # fuck this is strange. but it's correct  (right?) 
            
            newTripletIdx[k,0] = i 
            newTripletIdx[k,1] = targetTmplIdx[i] + bNumTrain 
            newTripletIdx[k,2] = minDstOtherClassTmplIdx + bNumTrain
            k += 1
            
            if k >= batchTripletIdxEnd:
                break
            
            #--- closets wrong from same class
            
            # set target tmpl to have a high distance, such that it wont be selected as the closest
            farAwayDst = numpy.max(dst[i,:])+1
            dst[i,targetTmplIdx[i]] = farAwayDst
            # also exclude those with same pose (and same class)
            trSampIdx = sampIdx[i + nBatch*batchSize]
            sameClassTmplSampIdx = sampIdx[nBatch*batchSize + bNumTrain + bSameClassTmplIdx[sLabel]]  # orig sample indizes of the templates of the same class
            trRot = trainRots[sLabel,trSampIdx]
            scTmplRots = tmplRots[sLabel,sameClassTmplSampIdx]
            sims = numpy.dot(scTmplRots,trRot.T) # similarities of poses of training sample i and  
            if zRotInv[sLabel] == 2:
                sims = numpy.maximum(sims, numpy.dot(scTmplRots,(trRot*numpy.array([-1,-1,1])).T) )
                
            numEqualSameClassTmpls = (sims >= 1.0).sum()
            if  numEqualSameClassTmpls < len(bSameClassTmplIdx[sLabel]): 
                if numEqualSameClassTmpls > 1:
                    tooSimIdx = (sims >= 1.0).nonzero()[0]
    #                 print("tooSimIdx {}".format(tooSimIdx))
    #                 print("dst before")
    #                 print(dst[i])
    #                 print("dst.shape {}".format(dst.shape))
    #                 print("bSameClassTmplIdx[sLabel][tooSimIdx] {}".format(bSameClassTmplIdx[sLabel][tooSimIdx]))
                    dst[i,bSameClassTmplIdx[sLabel][tooSimIdx]] = farAwayDst
    #                 print("sLabel {}".format(sLabel))
    #                 print("targetTmplIdx[i]-off {}".format(targetTmplIdx[i]-(tmplStartIdx[sLabel]-bNumTrain)))
    #                 print("targetTmplIdx[i] {}".format(targetTmplIdx[i]))
    #                 print(sims)
    #                 print(trRot)
    #                 print(scTmplRots)
    #                 print("dst after")
    #                 print(dst[i])
    #                 raise ValueError("die here and rest in peace")
            
            
                minDstSameClassTmplIdx = numpy.argsort( dst[i,bSameClassTmplIdx[sLabel]] )
                minDstSameClassTmplIdx = minDstSameClassTmplIdx[numpy.minimum( tmplListIdx, len(minDstSameClassTmplIdx)-1 )]
                minDstSameClassTmplIdx = bSameClassTmplIdx[sLabel][ minDstSameClassTmplIdx ]
                
                newTripletIdx[k,0] = i 
                newTripletIdx[k,1] = targetTmplIdx[i] + bNumTrain
                newTripletIdx[k,2] = minDstSameClassTmplIdx + bNumTrain
                k += 1
                            
            else:
                # all tmpls of the same class have the same pose as the training sample.
                #  -> for now, just replicate the last one (closest from other class)
                newTripletIdx[k,0] = i 
                newTripletIdx[k,1] = newTripletIdx[k-1,1] 
                newTripletIdx[k,2] = newTripletIdx[k-1,2]
                k += 1
                

            i += 1 
            if i >= bNumTrain:  # loop over training samples
                i = 0
                # when all training samples where considered, take the next best (second/third/... closest tmpl) as "far" tmpl
                tmplListIdx += 1

    train_set.tripletIdx = newTripletIdx
    train_set.nTripletsPerBatch = newNTripletsPerBatch
    
        
    
def checkDataset(train_set):
    
    batchSize = train_set.batchSize
    numSamples = train_set.x.shape[0]
    nBatches = numSamples / batchSize
    assert nBatches*batchSize == train_set.x.shape[0], "number of samples {} not divisible by batchSize {}".format(numSamples,batchSize) 
    
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
    
    nSamples = data_set.x.shape[0]
    nBatches = nSamples / data_set.batchSize
    batchSizeBytes = data_set.x.nbytes / nBatches
    print("  nTrainSamples: {}".format(nSamples))
    print("  nTrainBatches: {}".format(nBatches))
    print("  batchSize: {}".format(data_set.batchSize))
    print("  batchSizeMBytes: {}mb".format(batchSizeBytes/(1024.*1024.)))
                
    
    
def makeTrainsetSizeAMultipleOfValsetSize(train_set,val_set):
    nTrainSamples = train_set.x.shape[0]
    nTrainBatches = nTrainSamples / train_set.batchSize
    #batchSizeBytes = train_set.x.nbytes / nTrainBatches

    nValSamples = val_set.x.shape[0]
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
                
        nTrainSamples = train_set.x.shape[0]
        nTrainBatches = nTrainSamples / train_set.batchSize

        nValSamples = val_set.x.shape[0]
        nValBatches = nValSamples / val_set.batchSize
        print("--- valset   nsamp {},  batchsize {}".format(nValSamples,val_set.batchSize))
        print("--- trainset nsamp {},  batchsize {}".format(nTrainSamples,train_set.batchSize))

    return train_set, val_set

    #macroBatchSize = nTrainBatches






def linemod_train_realandsynth_main(cfg):
    
    v = sys.version_info
    print('running on python {}.{}.{}'.format(v.major,v.minor,v.micro))
    
    #rng = numpy.random.RandomState(23455)
    rng = numpy.random.RandomState(12345)
    #theano.config.compute_test_value = 'warn'
    theano.config.exception_verbosity = 'high'
    
    #resPath = "../data/results/tmp/"
    resPath = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../data/results/linemod_realandsynth/") + "/" 
    resPath = readCfgParam(cfg,'paths', 'resPath',resPath)

    startTimePidStr = nowPidStr()
    
    #
    start_time = time.clock()
    
    showPlots = readCfgIntParam(cfg,'misc','showPlots',default=1) > 0

    #############################################################################
    print("create data")

    targetSize = readCfgIntParam(cfg,'input','targetSize',default=64)
    imgSizeW = targetSize
    imgSizeH = targetSize

    inputMode = readCfgIntParam(cfg, 'input','mode', default=0)
    modeStrs = ('dpt','img','dptimg')
    
    seqNamesStr = readCfgParam(cfg, 'input','objs', default="")
    seqNames = map(str.strip,seqNamesStr.split(",")) 
    zRotInvStr = readCfgParam(cfg, 'input','zRotInv', default=[])
    zRotInv = list(eval(zRotInvStr))
    nSeqs = len(seqNames)

    fixedNumTrain = readCfgIntParam(cfg, 'trainset','fixedNumTrain', default=1) # split this number of training from the test sequences
    
    numNoisyTrainCopies = readCfgIntParam(cfg, 'trainset','numNoisyTrainCopies', default=7)
    trainSplitFactor = readCfgFloatParam(cfg, 'trainset','trainSplitFactor', default=0.5)
    
    
    dataBasePath = readCfgParam(cfg, 'paths', 'dataBase',default='../data/')
    imgsPklFileName = '{}linemod_imgs_synth_and_{}test_norm_s{}_i{}_o{}.pkl'.format(dataBasePath,fixedNumTrain,targetSize,modeStrs[inputMode],nSeqs)
    trainSetPklFileName = '{}linemod_trainset_synth_and_{}test_s{}_{}_n{}_o{}.pkl'.format(dataBasePath,fixedNumTrain,targetSize,modeStrs[inputMode],numNoisyTrainCopies,nSeqs)

    lmDataBasepath = readCfgParam(cfg,'paths', 'lmDataBase',default='/home/wohlhart/work/data/linemod/')

    loadImgsFromPkl = True
    if (not loadImgsFromPkl) or (not os.path.isfile(imgsPklFileName)):
        if loadImgsFromPkl:
            print("Couldn't load from {}".format(imgsPklFileName))
        print("Loading images from files ... ")
        # LOAD from images AND SAVE to pkl
        load_start_time = time.clock()
        tmplSeqs,trainSeqs,testSeqs = loadData(lmDataBasepath,targetSize,inputMode,trainSplitFactor=trainSplitFactor,fixedNumTrain=fixedNumTrain,seqNames=seqNames,zRotInv=zRotInv)
        load_end_time = time.clock()
        
    
        f = open(imgsPklFileName, 'wb')
        cPickle.dump((tmplSeqs,trainSeqs,testSeqs), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()          
        print ('Creating took %.1fs' % ((load_end_time - load_start_time)) )
        
    else:
        print("Loading images from pkl '{}' ... ".format(imgsPklFileName))
        # LOAD from pkl
        pklload_start_time = time.clock()
        f = open(imgsPklFileName, 'rb')
        tmplSeqs,trainSeqs,testSeqs = cPickle.load(f)
        f.close()  
        pklload_end_time = time.clock()
    
        print ('Loading took %.1fs' % ((pklload_end_time - pklload_start_time)) )
    

    # DEBUG    
#     patch0 = trainSeqs['bowl'].data[1]
#     print("patch0.filename {}".format(patch0.frame.filename))
#     cv2.imshow("trainPatch",(patch0.dpt+1.)/2.)
#     print("rcp {}".format(patch0.frame.pose.relCamPosZRotInv))
#     #print("rot {},tra {}".format(patch0.frame.pose.rot, patch0.frame.pose.tra))
#     cv2.waitKey(0)
#     raise ValueError("stop here")


    
    loadTrainSetFromPkl = True
    if loadTrainSetFromPkl and os.path.isfile(trainSetPklFileName):
        print ('Loading train_set from pkl {}'.format(trainSetPklFileName) )
        pklload_start_time = time.clock()
        f = open(trainSetPklFileName, 'rb')
        train_set,val_set = cPickle.load(f)
        f.close()  
        pklload_end_time = time.clock()
     
        if isinstance(train_set.x,str):
            with open(os.path.join(os.path.dirname(trainSetPklFileName),train_set.x),'rb') as f:
                train_set.x = numpy.load(f)        
        if isinstance(val_set.x,str):
            with open(os.path.join(os.path.dirname(trainSetPklFileName),val_set.x),'rb') as f:
                val_set.x = numpy.load(f)        
        
        print ('Loading train_set took %.1fs' % ((pklload_end_time - pklload_start_time)) )
         
         
    else:
        create_start_time = time.clock()
        
        train_set = None
        val_set = None
        
        batchSize = readCfgIntParam(cfg,'trainset','batchSize',default=300)
        #batchSize = 8
        
        numTrainPerClass = numpy.array([len(trainSeqs[seqN].data) for seqN in trainSeqs])
        if numpy.count_nonzero(numTrainPerClass - numTrainPerClass[0]) > 0:
            # cannot deal with unequal training sequence sizes ->
            # v1: cut to min
#             minTrainSeqSize = numpy.min(numTrainPerClass)
#             for seqN in trainSeqs:
#                 trainSeqs[seqN] = NamedImgSequence(seqN,trainSeqs[seqN].data[:minTrainSeqSize]) 
    
            # v2: repeat random frames to fill up shorter sequences
            numTrainPerClass = numpy.max(numTrainPerClass)
            for seqN in trainSeqs:
                if len(trainSeqs[seqN].data) < numTrainPerClass:
                    idx = numpy.random.choice(len(trainSeqs[seqN].data),numTrainPerClass-len(trainSeqs[seqN].data),replace=False)
                    for i in idx:
                        trainSeqs[seqN].data.append(trainSeqs[seqN].data[i])
     
        
        for i in range(numNoisyTrainCopies):
            trSeqs = copy.deepcopy(trainSeqs)
        
            # add noise ?
            #addNoiseToTrainSeqs(trSeqs,inputMode,rng,addSoftNoise=True,addHardNoise=False,addBgNoise=False)    
            addNoiseToTrainSeqs(trSeqs,inputMode,rng,addSoftNoise=True,softNoiseRange=0.01,addHardNoise=False,addBgNoise=True)
        
            # create dataset from img sequences
            if train_set is None:
                train_set = createTripleAndPairsDatasetFromImgSeq(tmplSeqs,trSeqs,inputMode=inputMode,batchSize=batchSize) 
                val_set = createTripleAndPairsDatasetFromImgSeq(tmplSeqs,trSeqs,inputMode=inputMode,batchSize=batchSize)
            else:
                ts = createTripleAndPairsDatasetFromImgSeq(tmplSeqs,trSeqs,inputMode=inputMode,batchSize=batchSize)
                train_set.append(ts)
                
                # no so much val data ...
                #vs = createTripleAndPairsDatasetFromImgSeq(tmplSeqs,trSeqs,inputMode=inputMode,batchSize=batchSize)
                #val_set.append(vs)

        create_end_time = time.clock()
        
        # DEBUG - need to check correctness of data
        checkDataset(train_set)
        checkDataset(val_set)
         
        # cannot pickle arrays larger than 2**32
        print("train_set.x.nbytes {},{}".format(train_set.x.nbytes,2**32))
        if train_set.x.nbytes > 2**31:
            trainSetXFilename = trainSetPklFileName + '.trainx'
            print("saving trainset x to extra file: '{}'".format(trainSetXFilename))
            with open(trainSetXFilename,'wb') as f:
                numpy.save(f,train_set.x)                
            tmpTrainX = train_set.x 
            train_set.x = os.path.basename(trainSetXFilename)
            print("train_set.x {}".format(train_set.x))
            print("train_set.__dict__ {}".format(train_set.__dict__))

            valSetXFilename = trainSetPklFileName + '.valx'
            print("saving valset x to extra file: '{}'".format(valSetXFilename))
            with open(valSetXFilename,'wb') as f:
                numpy.save(f,val_set.x)
            tmpValX = val_set.x 
            val_set.x = os.path.basename(valSetXFilename)
 
        f = open(trainSetPklFileName, 'wb')
        cPickle.dump((train_set,val_set), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()          
        
        if isinstance(train_set.x,str):
            train_set.x = tmpTrainX
            val_set.x = tmpValX
            
        print ('Creating took %.1fs' % ((create_end_time - create_start_time)) )
    
    
    
    numClasses = len(trainSeqs)

    #---------------------------------------------------------------
    # make trainset size a multiple of valset size
    #train_set, val_set = makeTrainsetSizeAMultipleOfValsetSize(train_set,val_set)

    nTrainSamples = train_set.x.shape[0]
    nTrainBatches = nTrainSamples / train_set.batchSize

    nValSamples = val_set.x.shape[0]
    nValBatches = nValSamples / val_set.batchSize
        
    #---------------------------------------------------------------
    
    macroBatchSize = readCfgIntParam(cfg,'trainset','macroBatchSize',default=-1)
    maxBytes = eval(readCfgParam(cfg,'trainset','maxBytes',default=50*1024*1024))  # 1 GB
    
    if macroBatchSize < 0:
        macroBatchSize = findNiceMacroBatchSize(macroBatchSize, maxBytes, train_set, nValBatches)
        
    macroBatchSize = adjustMacroBatchSizeToDivisorOfNumTrainBatchesAndNumValBatches(macroBatchSize, nTrainBatches, nValBatches)
    #---------------------------------------------------------------
    
    
    nTrainSamples = train_set.x.shape[0]
    nTrainBatches = nTrainSamples / train_set.batchSize
    #batchSizeBytes = train_set.x.nbytes / nTrainBatches

    #nValSamples = val_set.x.shape[0]
    #nValBatches = nValSamples / val_set.batchSize
    
    print("---------------------------------------")
    print("Trainset info:")
    printDatasetInfo(train_set)
    print("")
    print("Valset info:")
    printDatasetInfo(val_set)
    print("")
        
    #raise ValueError("ok") 

    #numpy.unique(train_set.y)
    
    # DEBUG - vis data
    #montageLinemodTrainSetSample(train_set)
    #cv2.waitKey()
    
    # TODO: vis sample and its closest tmpl
    #montageLinemodTrainSamplesAndTmpls(train_set)
    #cv2.waitKey()
    #raise ValueError("stop here")
    
    #visBatchedTrainDatasetCamPoses(train_set)
    #plt.show()
    #raise ValueError("stop here")
    
    #############################################################################
    print("create network")
    #descrNetType = 2
    #descrNetType = 3  # LeNet  
#     descrNetType = readCfgIntParam(cfg, 'net', 'type', default=5)
    outputDescrLen = readCfgIntParam(cfg, 'net', 'outputDescrLen', default=3)
#     
    if inputMode == 0:
        nChannels = 1 # just dpt
    elif inputMode == 1:
        nChannels = 3 # img
    else:
        nChannels = 4 # img+dpt
#         
#     descrNetParams = NetworkParams(descrNetType,nChan=nChannels,wIn=imgSizeW,hIn=imgSizeH,batchSize=train_set.batchSize,nOut=outputDescrLen)
#     descrNet = Network(rng,cfgParams=descrNetParams)

    inputDim = [train_set.batchSize,nChannels,imgSizeH,imgSizeW]
    descrNetParams = SimpleFFNetworkParams(inputDim)
    descrNetParams.initFromConfig(cfg, 'net')
    descrNetParams.debugPrint()
    assert outputDescrLen == descrNetParams.outputDim[1],"outputDescrLen set in the network config does not match what the defined net calculates"    
    
    descrNet = SimpleFFNetwork(rng,cfgParams=descrNetParams)

    # check distances between layer0 filters
    #checkFiltersDist(descrNet)
     
    pairsAvailable = train_set.pairIdx is not None
    descrNetTrainerParams = NetworkTrainingParams()
    descrNetTrainerParams.acceptPartialMacroBatches = False
    descrNetTrainerParams.batch_size = train_set.batchSize
    descrNetTrainerParams.use_triplets = True 
    descrNetTrainerParams.use_pairs = (readCfgIntParam(cfg, 'train', 'use_pairs', 1) == 1) and pairsAvailable 
    descrNetTrainerParams.learning_rate = readCfgFloatParam(cfg, 'train','learning_rate',default=0.01)
    descrNetTrainerParams.learning_rate_schedule_iter = readCfgIntParam(cfg,'train','learning_rate_schedule_iter',default=nTrainBatches*100)  # decrease after every n minibaches (every 100 epochs)
    descrNetTrainerParams.learning_rate_schedule_factor = readCfgFloatParam(cfg,'train','learning_rate_schedule_factor',default=0.9) # decrease by 0.9
    descrNetTrainerParams.save_traindescr_interval = readCfgIntParam(cfg,'train','save_traindescr_interval',default=0)  # decrease after every n minibaches (every 100 epochs)    
    descrNetTrainerParams.save_traindescr_filebase = readCfgParam(cfg,'train','save_traindescr_filebase',default='')  # decrease after every n minibaches (every 100 epochs)
    descrNetTrainerParams.momentum = readCfgFloatParam(cfg,'train','momentum',default=0.9)
    #descrNetTrainerParams.weightreg_factor = 1e-3#1e-7
    descrNetTrainerParams.weightreg_factor = readCfgFloatParam(cfg,'train','weightreg_factor',default=1.0) # 1e-4, 1e-2

    descrNetTrainerParams.tripletCostType = readCfgIntParam(cfg,'train','tripletCostType',default=6)  # 
    descrNetTrainerParams.tripletThresh = readCfgFloatParam(cfg,'train','tripletThresh',default=0.01)    
    #descrNetTrainerParams.tripletThresh = 0.001
    
    descrNetTrainerParams.nClasses = numClasses # only needed if use_labels = True
    
    descrNetTrainerParams.debugResPath = resPath
        
    descrNetTrainerParams.debugPrint()
    
    print("num weight in descrNet: {}".format(descrNet.getNumWeights()))
    print("    total: {}".format(numpy.sum([numpy.prod(ws) for ws in descrNet.getNumWeights()])))
    
    #raise ValueError("stop here")

    # get template descriptors
    
    print("setup trainer")
    descrNetTrainer = NetworkTrainer(descrNet,descrNetTrainerParams,rng)
    descrNetTrainer.setDataAndCompileFunctions(train_set,val_set,macroBatchSize,compileDebugFcts=True)#True)
    
    
    # take first few batches of training_data as eval data
    vis_eval_set = val_set
#    n = train_set.batchSize*8
#    si = {'rots':train_set.sampleInfo['rots'][0:n]}
#    vis_eval_set = BatchedLinemodTrainDataset(x=train_set.x[0:n],y=train_set.y[0:n],sampleInfo=si)
    #calcAndVis3DDescriptors(descrNet,vis_eval_set,resPath+startTimePidStr+"_descr_init.png")
    calcAndVis3DDescriptorsWithRotWithTrainer(descrNetTrainer,vis_eval_set,resPath+startTimePidStr+"_descr_init.png")
    
    #visualizeNetworkFilters(descrNet,resPath+startTimePidStr+"_filters_init.png")
    visNetworkFiltersMontage(descrNet,resPath+startTimePidStr+"_filters_init.png")
    if showPlots:
        plt.show(block=False) 
    
    ###################################################################
    #
    # TRAIN
    #
    nEpochs = readCfgIntParam(cfg, 'train', 'nEpochs', default=500)
    numBootstrappingRounds = readCfgIntParam(cfg, 'train', 'numBootstrappingRounds', default=3)
    nBootstrappingEpochs = readCfgIntParam(cfg, 'train', 'nBootstrappingEpochs', default=500)
    nFinetuneingEpochs = readCfgIntParam(cfg, 'train', 'nFinetuneingEpochs', default=500)
    
    nIter = 0
    
    train_start_time = time.clock()
    train_res = descrNetTrainer.train(n_epochs=nEpochs,storeFilters=False)
    train_end_time = time.clock()
    print ('>> Training took %.1fs'% ((train_end_time - train_start_time)) )        
    
    train_costs = train_res[0]
    #wvals = train_res[1]
    nIter += train_res[1]
    
    #calcAndVis3DDescriptors(descrNet,vis_eval_set,resPath+startTimePidStr+"_descr_trained0.png")    
    calcAndVis3DDescriptorsWithRotWithTrainer(descrNetTrainer,vis_eval_set,resPath+startTimePidStr+"_descr_trained0.png")
    
    #visualizeNetworkFilters(descrNet,resPath+startTimePidStr+"_filters.png")
    visNetworkFiltersMontage(descrNet,resPath+startTimePidStr+"_filters.png")

    # save intermediate descrNet        
    with open(resPath+startTimePidStr+"_b0_descrNet.pkl","wb") as f:
        cPickle.dump(descrNet,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(train_costs,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(wvals,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(descrNetTrainerParams,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(imgsPklFileName,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(trainSetPklFileName,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(cfg,f,protocol=cPickle.HIGHEST_PROTOCOL)

    ##############################################################
    # Bootstrap
    for i in xrange(numBootstrappingRounds):
        
        print("########### Bootstrapping, round {} ##########".format(i+1))
        
        bootstrapTriplets(descrNetTrainer,train_set)
        checkDataset(train_set)
        bootstrapTriplets(descrNetTrainer,val_set)
        checkDataset(val_set)
        
        descrNetTrainer.setDataAndCompileFunctions(train_set,val_set,macroBatchSize,compileDebugFcts=True)
        
        train_start_time = time.clock()
        train_res = descrNetTrainer.train(n_epochs=nBootstrappingEpochs,storeFilters=False,iterOffset=nIter)
        train_end_time = time.clock()
        print ('>> Training took %.1fs'% ((train_end_time - train_start_time)) )        
        
        nIter += train_res[2]
        
        train_costs.extend(train_res[0])
        
        calcAndVis3DDescriptorsWithRotWithTrainer(descrNetTrainer,vis_eval_set,resPath+startTimePidStr+"_descr_trained_b{}.png".format(i+1))
        
        #visualizeNetworkFilters(descrNet,resPath+startTimePidStr+"_filters_b{}.png".format(i+1))
        visNetworkFiltersMontage(descrNet,resPath+startTimePidStr+"_filters_b{}.png".format(i+1))

        # save intermediate descrNet        
        with open(resPath+startTimePidStr+"_b{}_descrNet.pkl".format(i+1),"wb") as f:
            cPickle.dump(descrNet,f,protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(train_costs,f,protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(wvals,f,protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(descrNetTrainerParams,f,protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(imgsPklFileName,f,protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(trainSetPklFileName,f,protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(cfg,f,protocol=cPickle.HIGHEST_PROTOCOL)
        
        
    ################################################
    # Finetune
    #
    if nFinetuneingEpochs > 0:
        descrNetTrainer.cfgParams.learning_rate *= 0.1    
        train_start_time = time.clock()
        train_res = descrNetTrainer.train(n_epochs=nFinetuneingEpochs,storeFilters=True,iterOffset=nIter)
        train_end_time = time.clock()
        print ('>> Training took %.1fs'% ((train_end_time - train_start_time)) )        
        
        train_costs.extend(train_res[0])
        nIter += train_res[2]
        
        calcAndVis3DDescriptorsWithRotWithTrainer(descrNetTrainer,vis_eval_set,resPath+startTimePidStr+"_descr_trained_final.png")
        #visualizeNetworkFilters(descrNet,resPath+startTimePidStr+"_filters_final.png")
        visNetworkFiltersMontage(descrNet,resPath+startTimePidStr+"_filters_final.png")
    
    #################################################
    # save stuff
    with open(resPath+startTimePidStr+"_descrNet.pkl","wb") as f:
        cPickle.dump(descrNet,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(train_costs,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(wvals,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(descrNetTrainerParams,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(imgsPklFileName,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(trainSetPklFileName,f,protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(cfg,f,protocol=cPickle.HIGHEST_PROTOCOL)
    
    # TODO write config/params as cfg/ini/xml - file as well so we dont depend on the pickle (if the pickle cant be loaded anymore and we still what to see the config that led to the results)
    
    
    ##################
    end_time = time.clock()
    print ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs'% ((end_time - start_time)) )        
    print("done")
    
    if showPlots:
        plt.show(block=True)

        ch = 0
        while ((ch & 0xFF) != ord('q')) and (ch >= 0):
            print("waiting for 'q'")
            ch = cv2.waitKey() 
            print(" ... got '{}' ({})".format(chr(ch & 0xFF),ch))
    
    cv2.destroyAllWindows()
    
    
    
if __name__ == '__main__':
    
    raise ValueError("this is not finished. it should call train and then test, but from the other modules")
    
    if len(sys.argv) > 1:
        cfgFiles = sys.argv[1:]
    else:
        cfgFiles = ['config.ini']

    #assert os.path.isfile(cfgFile), "No config file found at '{}'".format(cfgFile)
    
    cfgparser = ConfigParser.SafeConfigParser()
    res = cfgparser.read(cfgFiles)
    if len(res) == 0:
        #raise ValueError("None of the config files could be read")
        print "Error: None of the config files could be read"
        sys.exit(1)
    
    linemod_train_realandsynth_main(cfgparser)
    
