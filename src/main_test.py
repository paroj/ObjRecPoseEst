'''
Created on Apr 30, 2015

@author: wohlhart
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # @UnresolvedImport @UnusedImport
from util.matplotlib2tikz import save as tikz_save

import time, os, glob, gc, cPickle, sys

import theano
import numpy
import cv2
import scipy.spatial.distance
import scipy.linalg

from ConfigParser import SafeConfigParser


from tnetcore.network import NetworkParams, Network
from tnetcore.trainer import NetworkTrainingParams, NetworkTrainer
from tnetcore.util import readCfgIntParam, readCfgParam

from util.vis import showLinemodFrame, calcAndVis3DDescriptorsWithRotWithTrainer,\
    visBatchedDatasetCamPoses, visBatchedTrainDatasetCamPoses, plotAngleErrors,\
    visClosestTmpls, visWrongestClosestTmpls, visSimsVsDsts,\
    calcAndVis3DDescriptorsWithRot
from data.importers import LinemodTrainDataImporter
from data.batcheddatasets import BatchedImgSeqDataset
from main_train import loadTrainResultsPickle, findNiceMacroBatchSize
from tnetcore.data import DataManager, MacroBatchManagerConfig,\
    MacroBatchManager, BatchedDataset
import collections


def debugTestDataset(lmDataset):

    seq0 = lmDataset.imgSeqs[0]
    print("type(seq0) {}".format(type(seq0)))
    
    #for frame in seq0.data:
    #    print(frame.filename)
    
    # Test Frame
    #frame = seq0.data[47]
    frame = seq0.data[0]

    cv2.imshow("image",frame.img+0.5)
    cv2.imshow("dpt",frame.dpt/400.)  # original data in cm, in float32 -> so we have to scale down to see it

    print("frame.pose {}".format(frame.pose))

    # find center point of object in image
    showLinemodFrame(frame)
    
    

    
    

def loadTmplSeqsBatchedDataset(lmDataBasepath ,targetSize = 64, batchSize = 300, inputMode = 0, seqNames = None, zRotInv = None):
    
    #floatX = theano.config.floatX  # @UndefinedVariable

    if seqNames is None:    
        seqNames = ("ape","bowl","cat","duck")
        #zRotInv = (False,True,False,False) 
        zRotInv = (0, 1, 0, 0)

    linemodTmplDataPklFilename = "../data/linemod_tmpldata_o{}_i{}_s{}_bs{}.pkl".format(len(seqNames),inputMode,targetSize,batchSize)
    
    loadFromPkl = True
    if loadFromPkl and (os.path.isfile(linemodTmplDataPklFilename)):
        start_time = time.clock()
        with open(linemodTmplDataPklFilename,'rb') as f:
            tmpldata_set = cPickle.load(f)        
        end_time = time.clock()
        print ('Loading tmpl dataset from pkl took %.1fs' % ((end_time - start_time)) )
    else:
        if (loadFromPkl) and (not os.path.isfile(linemodTmplDataPklFilename)):
            print("Couldn't load from pkl '{}'".format(linemodTmplDataPklFilename)) 
        
        print("Loading tmpls from training images")
                
        #TODO REFACTOR: thow following could be move into LinemodTmplDataset
        
        imgNumsTemplates = numpy.arange(301)
        #imgNumsTemplates = numpy.arange(3)
        
        lmti = LinemodTrainDataImporter(lmDataBasepath)
        # TEMPLATES  
        #   * load image sequences
        #   * create batched dataset
        imgSeqs = []
        #totalNumImgs = 0
        for i in xrange(len(seqNames)):
            seqN = seqNames[i]
            zri = zRotInv[i]
            imgSeq = lmti.loadSequence(objName=seqN,imgNums=imgNumsTemplates,cropSize=20.0,targetSize=targetSize,zRotInv=zri)
            imgSeqs.append(imgSeq)
            #totalNumImgs += len(imgSeq.data)
        
        
        #TODO REFACTOR: from here on this should move into the BatchedTmplDataset
        tmpldata_set = BatchedImgSeqDataset()
        tmpldata_set.initFromImgSeqs(imgSeqs,inputMode=inputMode,batchSize=batchSize)
        
#         imgD0 = dataArrayFromPatch(imgSeqs[0].data[0],inputMode)
#         nChan,patchH,patchW = imgD0.shape
#         print("nChan, patchH, patchW {},{}".format(nChan,patchH,patchW))
#         
#         #nChan = getNChanFromInputMode(inputMode)
#         numBatches = numpy.ceil(float(totalNumImgs)/float(batchSize))
#         allData = numpy.zeros((numBatches*batchSize,nChan,patchH,patchW),dtype=floatX)
#         allLabels = -numpy.ones((numBatches*batchSize,),dtype=numpy.int32)  # unused space in last minibatch must be filled with label=-1
#         allRots = numpy.zeros((numBatches*batchSize,3),dtype=numpy.float32)
#                 
#         k = 0    
#         for i in xrange(len(imgSeqs)):
#             for patch in imgSeqs[i].data:
#                 imgD = dataArrayFromPatch(patch,inputMode)
#                 allData[k] = imgD
#                 allLabels[k] = i
#                 allRots[k] = patch.frame.pose.relCamPosZRotInv
#                 k += 1
#         
#         
#         sampleInfo = {'maxSimTmplIdx': None, 
#                       'sampIdx': numpy.arange(numBatches*batchSize), 
#                       'tmplBatchDataStartIdx': None,  
#                       'rots': allRots, 
#                       'tmplRots': None, 
#                       'trainRots': None }
#                         
#         tmpldata_set = BatchedDataset(x=allData, y=allLabels, sampleInfo=sampleInfo, batchSize=batchSize)
        
        # save tmpl dataset
        with open(linemodTmplDataPklFilename,'wb') as f:
            cPickle.dump(tmpldata_set,f,protocol=cPickle.HIGHEST_PROTOCOL)

    return tmpldata_set


def getAccRel(sim,dst):
    # for how many is the most similar one also the one with the smallest dst
    #  relaxed formulation: of the most similar ones (if more than one is equally similar) one is among those with smallest dst (if more than one is equally far away)
    maxSim = numpy.max(sim, axis=1)
    minDst = numpy.min(dst, axis=1)
    nSamp = sim.shape[0]
    nCorrect = 0.
    nCorrectClass = 0.
    for i in xrange(nSamp):
        maxSimIdx, = (sim[i,:] == maxSim[i]).nonzero()
        minDstIdx, = (dst[i,:] == minDst[i]).nonzero()
        if len(numpy.intersect1d(maxSimIdx,minDstIdx, assume_unique=True)) > 0:
            nCorrect += 1.
            
        if numpy.min(sim[i,minDstIdx]) > -2.:
            nCorrectClass += 1.
            
    acc = nCorrect / nSamp
    
    # classification accuracy. for how many percent is the closest from the correct class
    classAcc = nCorrectClass / nSamp
     
    return (acc,classAcc)

def getAcc(sim,dst):
    # for how many is the most similar one also the one with the smallest dst
    maxSim = numpy.max(sim, axis=1)
    minDstIdx = numpy.argmin(dst, axis=1)
    nSamp = sim.shape[0]
    nCorrect = 0.
    for i in xrange(nSamp):
        maxSimIdx = sim[i,:] == maxSim[i]
        if len(numpy.intersect1d(maxSimIdx,minDstIdx[i], assume_unique=True)) > 0:
            nCorrect += 1.
    return nCorrect / nSamp



def getClassSepScore(sim, dst):
    # TODO: class separation score. for each sample: relation dst to it's target tmpl (or any other closest from same class) vs dst to closest from other classes
    
    nSamp = sim.shape[0]
    perSampSepScore = numpy.zeros(nSamp)
    for i in xrange(nSamp):
        sameClassIdx = (sim[i] > -2).nonzero()[0]
        otherClassIdx = (sim[i] <= -2).nonzero()[0]
        # dst to closest from same class vs from other class
        scDst = numpy.min(dst[i][sameClassIdx])
        dcDst = numpy.min(dst[i][otherClassIdx])
        if scDst == 0:
            perSampSepScore[i] = -1
        else:
            perSampSepScore[i] = dcDst / scDst
            
    maxSep = numpy.max(perSampSepScore)
    for i in xrange(nSamp):
        if perSampSepScore[i] < 0:
            perSampSepScore[i] = maxSep+1
    
    return perSampSepScore



def plotPerSampSepScore(perSampSepScore,fileName=None,tikzFileName=None,pklFileName=None):
    
    perSampSepScore = numpy.minimum(perSampSepScore,4.0)
    h,edges = numpy.histogram(perSampSepScore, bins=60)
    binWidth = edges[1]-edges[0]
    binCenters = (edges[:-1]+edges[1:])/2.0
    barWidth = binWidth*0.9
    fig = plt.figure()
    plt.bar(binCenters-barWidth/2,h,width=barWidth,log=True)
    
    if fileName is not None:
        if isinstance(fileName,list):
            for fn in fileName:
                plt.savefig(fn)
        else:  
            plt.savefig(fileName)            

    if tikzFileName is not None:
        tikz_save(tikzFileName, fig, figurewidth='\\figurewidth', figureheight='\\figureheight',show_info=False)

    if pklFileName is not None:
        with open(pklFileName,'wb') as f:
            data = {'perSampSepScore':perSampSepScore,'h':h,'edges':edges,'binWidth':binWidth,'binCenters':binCenters}
            cPickle.dump(data,f,protocol=cPickle.HIGHEST_PROTOCOL)


def linemod_test_main(fileName=None,cfg=None):
    
    theano.config.exception_verbosity = 'high'
    
    
    if fileName is None:
        g = glob.glob('../data/results/tmp/*_descrNet.pkl')
        #g = glob.glob('../data/results/cuda03_tmp/*_descrNet.pkl')
        fileName = sorted(g)[-1] 
        
    if os.path.isdir(fileName):
        g = glob.glob('{}/*_descrNet.pkl'.format(fileName))
        fileName = sorted(g)[-1]     
    
    outFNBase = fileName[:-len("descrNet.pkl")] + 'test'
    print("outFNBase {}".format(outFNBase))
    

    showPlots = readCfgIntParam(cfg,'misc','showPlots',default=1) > 0
    
    ######## load net
    trainRes = loadTrainResultsPickle(fileName)
    descrNet = trainRes[0]
    cfg = trainRes[2]
    descrNetTrainerParams = trainRes[3]
    #
        
    imgsPklFileName = readCfgParam(cfg, 'input', 'imgsPklFileName', default="")
    trainSetPklFileName = readCfgParam(cfg, 'train', 'trainSetPklFileName', default="")
    valSetPklFileName = readCfgParam(cfg, 'train', 'valSetPklFileName', default="")
    
    #-------------
    inputDim = descrNet.cfgParams._inputDim
    #if len(inputDim) > 1 and len(inputDim[0]) > 1:
    if isinstance(inputDim[0],collections.Iterable):
        inputMode = 3       # IMG/DPT
        inputSize = inputDim[0][2]
        
        # check
        heights = [idim[2] for idim in inputDim]
        widths = [idim[3] for idim in inputDim]
        assert all([w == inputSize for w in widths]) and all([h == inputSize for h in heights]), "can only handle quadratic inputs"     
    else:
        nChan = inputDim[1]
        if nChan == 1:
            inputMode = 0   # DPT
        elif nChan == 3:
            inputMode = 1   # IMG
        elif nChan == 4:
            inputMode = 2   # IMG + DPT
        else:
            raise ValueError("Unknown input mode: #channels = {}".format(nChan))
        inputSize = inputDim[2]
    
    print("inputMode = {}".format(inputMode))
    print("inputSize = {}".format(inputSize))

    print("num weight is descrNet: {}".format(descrNet.getNumWeights()))
    print("    total: {}".format(numpy.sum([numpy.prod(ws) for ws in descrNet.getNumWeights()])))

    ######## prepare test data    
    
    
    #lmDataBasepath = '/home/wohlhart/work/data/linemod/'
    lmDataBasepath = readCfgParam(cfg,'paths', 'lmDataBase',default='/home/wohlhart/work/data/linemod/')
    
    seqNamesStr = readCfgParam(cfg, 'input','objs', default="")
    seqNames = map(str.strip,seqNamesStr.split(",")) 
    zRotInvStr = readCfgParam(cfg, 'input','zRotInv', default=[])
    zRotInv = list(eval(zRotInvStr))
    
    print "trainset: {}".format(trainSetPklFileName)
    print "images: {}".format(imgsPklFileName)
    
    ###############################################################################################
    # create test set from test seq that was stored by main_linemod_realandsynth
    #   which did the split of the linemod data into train and test
    print("Loading images from pkl ... ")
    # LOAD from pkl
    pklload_start_time = time.clock()
    with open(imgsPklFileName, 'rb') as f:
        _,_,testSeqs = cPickle.load(f)
    f.close()  
    pklload_end_time = time.clock()

    print ('Loading took %.1fs' % ((pklload_end_time - pklload_start_time)) )
    
#     patch0 = testSeqs['cat'].data[52]
#     print("fn {}".format(patch0.frame.filename))
#     print("cn {}".format(patch0.frame.className))
#     cv2.imshow("cat52",patch0.dpt/2.+1.)
#     cv2.waitKey()
    
    #print("###### bs {}".format(descrNet.cfgParams.batch_size))
    testdata_set = BatchedImgSeqDataset()
    testdata_set.initFromImgSeqs(testSeqs, inputMode=inputMode, batchSize=descrNet.cfgParams.batch_size)
    
    
    # test it
    img0 = numpy.concatenate([x[0] for x in testdata_set.x],axis=0) if isinstance(testdata_set.x,list) else testdata_set.x[0]
    img0 = numpy.swapaxes(numpy.swapaxes(img0,0,1),1,2)
    print(img0.shape)
    if showPlots:
        cv2.imshow("img0",img0+0.5)
    #cv2.waitKey()
    

    # DEBUG: show cam poses
    #visBatchedDatasetCamPoses(testdata_set)
    #plt.show()
    #raise ValueError("stop here")
        

    #############################################################
    # Templates  dataset
    #
    tmpl_set = loadTmplSeqsBatchedDataset(lmDataBasepath,targetSize=inputSize,batchSize=descrNet.cfgParams.batch_size,inputMode=inputMode,seqNames=seqNames,zRotInv=zRotInv)
    assert isinstance(tmpl_set.x,list) or (tmpl_set.x.shape[1] == nChan)  # if it's a list, it's more complicated :(


    #############################################################
    #  general informations about tmpl and test datasets
    #
    uLab = numpy.setdiff1d(numpy.union1d(testdata_set.y,tmpl_set.y),numpy.array(-1),assume_unique=True)
    print("uLab {}".format(uLab))
    numClasses = len(uLab)
    
    testDataValidIdx, = (testdata_set.y >= 0).nonzero()
    tmplDataValidIdx, = (tmpl_set.y >= 0).nonzero()
    
    testLabels = testdata_set.y[testDataValidIdx]
    tmplLabels = tmpl_set.y[tmplDataValidIdx]
    
    nTestSamp = len(testDataValidIdx)
    nTmpls = len(tmplLabels)

    numTmplPerClass = nTmpls / numClasses
    print("numTmplPerClass {}".format(numTmplPerClass))
    print("tmplDataValidIdx {}".format(len(tmplDataValidIdx)))
    
    tmplRots = tmpl_set.sampleInfo['rots'][tmplDataValidIdx]
    print("tmplRots.shape {}".format(tmplRots.shape)) 
    tmplRots = tmplRots / numpy.tile(numpy.linalg.norm(tmplRots,axis=1).reshape(nTmpls,1),(1,3))  # normalize

    testRots = testdata_set.sampleInfo['rots'][testDataValidIdx]
    testRots = testRots / numpy.tile(numpy.linalg.norm(testRots,axis=1).reshape(nTestSamp,1),(1,3))  # normalize
    print("tmplRots.shape {}, testRots.shape {}".format(tmplRots.shape,testRots.shape))    
    
    #################################################################################
    # create a dataManager
    dataManagerConfig = MacroBatchManagerConfig()
    dataManagerConfig.batch_size = testdata_set.batchSize

    macroBatchSize = readCfgIntParam(cfg,'trainset','macroBatchSize',default=-1)
    maxBytes = eval(readCfgParam(cfg,'trainset','maxBytes',default=50*1024*1024))
    nTestBatches = testdata_set.numSamples / testdata_set.batchSize
    nTmplBatches = tmpl_set.numSamples / tmpl_set.batchSize
    batchSizeBytes = testdata_set.dataSizeNBytes / nTestBatches 
    macroBatchSize = findNiceMacroBatchSize(macroBatchSize, maxBytes, batchSizeBytes, [nTestBatches,nTmplBatches])
    
    dataManager = MacroBatchManager(dataManagerConfig,macroBatchSize,acceptPartialMacroBatches=False)
    dataManager.setupVariables(testdata_set, doBorrow=True)
    
    #################################################################################
    # visualize testdata and traindata
    visTestDataDescrs = True
    testOnTrainData = False
    if testOnTrainData or visTestDataDescrs:
        
        # LOAD DATA
        print("Load train_set: {}".format(trainSetPklFileName))
        pklload_start_time = time.clock()
        #f = open(trainSetPklFileName, 'rb')
        #train_set,val_set = cPickle.load(f)
        #f.close()
        
        train_set = BatchedDataset.loadFromPkl(trainSetPklFileName)
        #val_set = BatchedDataset.loadFromPkl(valSetPklFileName)
          
        pklload_end_time = time.clock()

        
        print ('Loading train_set took %.1fs' % ((pklload_end_time - pklload_start_time)) )

        if visTestDataDescrs:
            calcAndVis3DDescriptorsWithRot(descrNet,testdata_set,dataManager,fileName=outFNBase+'_testdata_descrs.png')
            plt.title('test data')

        if testOnTrainData:
            #descrNetTrainer.test_on_train()
            calcAndVis3DDescriptorsWithRot(descrNet,train_set,dataManager,fileName=outFNBase+'_traindata_descrs.png')
            plt.title('train data')
   
    # test if the classes are mapped correctly by showing one sample of each class from train and test
#     for cl in range(4):
#         idx = numpy.flatnonzero(testdata_set.y == cl)
#         img = testdata_set.x[idx[0]].squeeze()
#         cv2.imshow("test{}".format(cl),img)
#         idx = numpy.flatnonzero(train_set.y == cl)
#         img = train_set.x[idx[0]].squeeze()
#         cv2.imshow("train{}".format(cl),img)
    
        #rng = numpy.random.RandomState(23455)
        #descrNetTrainer = DescrNetTrainer(descrNet,descrNetTrainerParams,rng)
        #descrNetTrainer.setDataAndCompileFunctions(train_set,val_set,compileDebugFcts=False)#True)
        #descrNetTrainer.setData(train_set,val_set,macroBatchSize=1)
        #descrNetTrainer.test_on_train()
        
    ##
    # vis cam poses for all data
    visCamPoses = False
    if visCamPoses and showPlots:
        visBatchedDatasetCamPoses(testdata_set)
        visBatchedDatasetCamPoses(train_set)
        
        visBatchedTrainDatasetCamPoses(train_set)
        plt.show(block=False)

    
    #raise ValueError("stop here")
    
    
    ##############################
    # TEST results 
    descr_comp_start_time = time.clock()
    res = descrNet.computeDescriptors(testdata_set,dataManager,batch_size=testdata_set.batchSize)
    descr_comp_end_time = time.clock()
    # throw away results for invalid samples (that were added to fill up minibatches)
    testDescrs = res[testDataValidIdx]

    print ('Computing descriptors for %d samples took %.1fs' % (testdata_set.numSamples,(descr_comp_end_time - descr_comp_start_time)) )
    
    # calc descrs for templates   
    #res = descrNet.computeDescriptors(tmpl_set.x,batch_size=tmpl_set.batchSize)
    dataManager.prepareForNewData()
    res = descrNet.computeDescriptors(tmpl_set,dataManager,batch_size=tmpl_set.batchSize)
    tmplDescrs = res[tmplDataValidIdx]

    print("tmplDescrs.shape {}".format(tmplDescrs.shape))
    
    # distances of descriptors 
    dst = scipy.spatial.distance.cdist(testDescrs,tmplDescrs)
    print("dst.shape {}".format(dst.shape))
    
    # calculate similarity of poses of train sample and templates of the sample class 
    sims = []
    for lab in uLab:        
        sim = numpy.dot(testRots[testLabels==lab],tmplRots[tmplLabels==lab].T)
        if zRotInv[lab] == 2:
            sim = numpy.maximum(sim,numpy.dot(testRots[testLabels==lab]*numpy.array([-1,-1,1]),tmplRots[tmplLabels==lab].T))
        sims.append(sim+2.)
    sim = scipy.linalg.block_diag(*sims) - 2.
    print("sim.shape {}".format(sim.shape))
    
    print("sim mn/mx {},{}".format(numpy.min(sim),numpy.max(sim)))
    
    # show sim matrix
    #cv2.imshow("sims",(sim-numpy.min(sim))/(numpy.max(sim) - numpy.min(sim)))
    
    # TODO, eval metrics:
    #    * pure accuracy: closest tmpl is the correct one (target) 
    #    * cmc curve (target tmpl is among the first k closest)
    #    * curve: number of samples for which the closest tmpl has a rotation error below x  
    
    # make sim block diag
    
    #------    
    (acc,classAcc) = getAccRel(sim,dst)
    print("total test acc: {}%".format(acc*100.))
    print("total test class acc: {}%".format(classAcc*100.))
    
    # perSampSepScore
    perSampSepScore = getClassSepScore(sim,dst)
    minSepScore = numpy.min(perSampSepScore)
    maxSepScore = numpy.max(perSampSepScore)
    meanSepScore = numpy.mean(perSampSepScore)
    medSepScore = numpy.median(perSampSepScore)
    print("perSampSepScore mn/mx/mean/med {},{},{},{}".format(minSepScore,maxSepScore,meanSepScore,medSepScore))
    
    tikzFN = outFNBase+'_class_sep.tikz'
    pdfFN = outFNBase+'_class_sep.pdf'
    pngFN = outFNBase+'_class_sep.png'
    pklFN = outFNBase+'_class_sep.pkl'
    plotPerSampSepScore(perSampSepScore,fileName=[pngFN,pdfFN],tikzFileName=tikzFN,pklFileName=pklFN)
    if showPlots:
        plt.show(block=False)  
    
    tikzFN = outFNBase+'_acc_vs_angleerror.tikz'
    pdfFN = outFNBase+'_acc_vs_angleerror.pdf'
    pngFN = outFNBase+'_acc_vs_angleerror.png'
    pklFN = outFNBase+'_acc_vs_angleerror.pkl'
    print("saving {}".format(pngFN))
    plotAngleErrors(sim,dst,fileName=[pdfFN,pngFN],tikzFileName=tikzFN,pklFileName=pklFN,showPlot=showPlots)
    
#     # vis min dst and max sim
#     minDst = numpy.min(dst,axis=1)
#     cv2.imshow("minDst",(dst == minDst).astype(numpy.float))
#     maxSim = numpy.min(sim,axis=1)
#     cv2.imshow("maxSim",(sim == maxSim).astype(numpy.float))

    visClosestTmpls(testdata_set,tmpl_set,dst,fileName=outFNBase+'_samples.png',showPlot=showPlots)

    visWrongestClosestTmpls(testdata_set,tmpl_set,dst,sim,fileName=outFNBase+'_wrongestsamples.png',showPlot=showPlots)
    
    ##
    # scatter all sims vs dsts per class
    tikzFN = outFNBase+'_sim_vs_dst.tikz'
    #pdfFN = outFNBase+'_sim_vs_dst.pdf'
    pngFN = outFNBase+'_sim_vs_dst.png'
    #visSimsVsDsts(sim=sim,dst=dst,uLab=uLab,rowLabels=testLabels,colLabels=tmplLabels,classLabels=seqNames,fileName=[pdfFN,pngFN],tikzFileName=tikzFN)#,mxDst=0.7)
    visSimsVsDsts(sim=sim,dst=dst,uLab=uLab,rowLabels=testLabels,colLabels=tmplLabels,classLabels=seqNames,fileName=pngFN)#,mxDst=0.7)
    if showPlots:
        plt.show(block=False)

    tikzFN = outFNBase+'_sim_vs_dst_clipped.tikz'
    pngFN = outFNBase+'_sim_vs_dst_clipped.png'
    #visSimsVsDsts(sim=sim,dst=dst,uLab=uLab,rowLabels=testLabels,colLabels=tmplLabels,classLabels=seqNames,fileName=[pdfFN,pngFN],tikzFileName=tikzFN)#,mxDst=0.7)
    visSimsVsDsts(sim=sim,dst=dst,uLab=uLab,rowLabels=testLabels,colLabels=tmplLabels,classLabels=seqNames,clip=25,fileName=pngFN)#,mxDst=0.7)
    if showPlots:
        plt.show(block=False)
        
    # save config to file
    targetCfgFile = outFNBase+'_cfg.ini'
    cfg.set('net', 'weightsshapes', "{}".format(descrNet.getNumWeights()))
    cfg.set('net', 'totalnumweights', "{}".format(numpy.sum([numpy.prod(ws) for ws in descrNet.getNumWeights()])))
    if not cfg.has_section('results'):
        cfg.add_section('results')
    cfg.set('results', 'acc', "{}".format(acc))
    cfg.set('results', 'classAcc', "{}".format(classAcc))
    cfg.set('results', 'minSepScore', "{}".format(minSepScore))
    cfg.set('results', 'maxSepScore', "{}".format(maxSepScore))
    cfg.set('results', 'meanSepScore', "{}".format(meanSepScore))
    cfg.set('results', 'medSepScore', "{}".format(medSepScore))
    with open(targetCfgFile, 'wb') as configfile:
        cfg.write(configfile)
        
    print("Saveing cfg along with other stuff to:")
    print("  '{}'".format(targetCfgFile))
    
    ######################################
    # wait for quit
    
    if showPlots:
        ch = 0
        plt.show(block=False)
        print("waiting for 'q'")
        while ((ch & 0xFF) != ord('q')) and (ch >= 0):

            ch = cv2.waitKey() 
            print(" ... got '{}' ({})".format(chr(ch & 0xFF),ch))
        
        cv2.destroyAllWindows()


    if (showPlots):
        print("showing plots")
    else:
        print("not showing plots")
    
if __name__ == '__main__':
    
    fileName = None
    if len(sys.argv)>1:
        #print(sys.argv[1])
        fileName = sys.argv[1]
    else:
        raise ValueError("Need one argument: the net to load")
    
    cfgparser = SafeConfigParser()
    if len(sys.argv)>2:
        res = cfgparser.read(sys.argv[2])
        if len(res) == 0:
            raise ValueError("None of the config files could be read")
    
    
    linemod_test_main(fileName,cfgparser)    
    
