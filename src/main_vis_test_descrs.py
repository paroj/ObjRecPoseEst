'''
Created on Apr 30, 2015

@author: wohlhart
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # @UnresolvedImport @UnusedImport

import time, os, glob, cPickle, sys

import theano
import numpy
import cv2

from ConfigParser import SafeConfigParser


from tnetcore.util import readCfgIntParam, readCfgParam

from util.vis import showLinemodFrame, calcAndVis3DDescriptorsWithRot
from data.batcheddatasets import BatchedImgSeqDataset
from main_train import loadTrainResultsPickle, findNiceMacroBatchSize
from tnetcore.data import MacroBatchManagerConfig, MacroBatchManager


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
    
  
    
    
def linemod_vis_test_descr_main(fileName=None,cfg=None):
    
    theano.config.exception_verbosity = 'high'
    
    if fileName is None:
        g = glob.glob('../data/results/tmp/*_descrNet.pkl')
        fileName = sorted(g)[-1] 
        
    if os.path.isdir(fileName):
        g = glob.glob('{}/*_descrNet.pkl'.format(fileName))
        fileName = sorted(g)[-1]     
    
    outFNBase = fileName[:-len("descrNet.pkl")] + 'test'
    print("outFNBase {}".format(outFNBase))
    
    showPlots = readCfgIntParam(cfg,'misc','showPlots',default=1) > 0
    
    ######## load net
    res = loadTrainResultsPickle(fileName)
    descrNet = res[0]
    cfg = res[2]
    descrNetTrainerParams = res[3]
    
    imgsPklFileName = readCfgParam(cfg, 'input', 'imgsPklFileName', default="")
    trainSetPklFileName = readCfgParam(cfg, 'train', 'trainSetPklFileName', default="")
    
    #-------------
    nChan = descrNet.cfgParams.layers[0].image_shape[1]
    if nChan == 1:
        inputMode = 0
    elif nChan == 3:
        inputMode = 1
    elif nChan == 4:
        inputMode = 2
    else:
        raise ValueError("Unknown input mode: #channels = {}".format(nChan))
    inputSize = descrNet.cfgParams.layers[0].image_shape[2]
    
    print("inputMode = {}".format(inputMode))
    print("inputSize = {}".format(inputSize))

    ######## prepare test data    
    lmDataBasepath = readCfgParam(cfg,'paths', 'lmDataBase',default='/home/wohlhart/work/data/linemod/')
    
    seqNamesStr = readCfgParam(cfg, 'input','objs', default="")
    seqNames = map(str.strip,seqNamesStr.split(",")) 
    zRotInvStr = readCfgParam(cfg, 'input','zRotInv', default=[])
    zRotInv = list(eval(zRotInvStr))
    
    ###############################################################################################
    # create test set from test seq that was stored by main_linemod_realandsynth
    #   which did the split of the linemod data into train and test
    print("Loading images from pkl ... ")
    # LOAD from pkl
    pklload_start_time = time.clock()
    with open(imgsPklFileName, 'rb') as f:
        _,_,testSeqs = cPickle.load(f)
    pklload_end_time = time.clock()

    print ('Loading took %.1fs' % ((pklload_end_time - pklload_start_time)) )
    
    testdata_set = BatchedImgSeqDataset()
    testdata_set.initFromImgSeqs(testSeqs, inputMode=inputMode, batchSize=descrNet.cfgParams.batch_size)
    

    #############################################################
    #  general informations about tmpl and test datasets
    #
    testDataValidIdx, = (testdata_set.y >= 0).nonzero()
    testLabels = testdata_set.y[testDataValidIdx]
    
    nTestSamp = len(testDataValidIdx)

    testRots = testdata_set.sampleInfo['rots'][testDataValidIdx]
    testRots = testRots / numpy.tile(numpy.linalg.norm(testRots,axis=1).reshape(nTestSamp,1),(1,3))  # normalize
    #print("tmplRots.shape {}, testRots.shape {}".format(tmplRots.shape,testRots.shape))    
    
    #################################################################################
    # create a dataManager
    dataManagerConfig = MacroBatchManagerConfig()
    dataManagerConfig.batch_size = testdata_set.batchSize

    macroBatchSize = readCfgIntParam(cfg,'trainset','macroBatchSize',default=-1)
    maxBytes = eval(readCfgParam(cfg,'trainset','maxBytes',default=50*1024*1024))
    nTestSamples = testdata_set.x.shape[0] 
    nTestBatches = nTestSamples / testdata_set.batchSize
    #nTmplSamples = tmpl_set.x.shape[0]
    #nTmplBatches = nTmplSamples / tmpl_set.batchSize
    batchSizeBytes = testdata_set.x.nbytes / nTestBatches 
    macroBatchSize = findNiceMacroBatchSize(macroBatchSize, maxBytes, batchSizeBytes, [nTestBatches])
    
    dataManager = MacroBatchManager(dataManagerConfig,macroBatchSize,acceptPartialMacroBatches=False)
    dataManager.setupVariables(testdata_set, doBorrow=True)
    
    #################################################################################
    # visualize testdata and traindata
        
    # LOAD DATA

    # create trainer for macrobatches descriptor computation
    #rng = numpy.random.RandomState(23455)
    #descrNetTrainer = NetworkTrainer(descrNet,descrNetTrainerParams,rng)
    #descrNetTrainer.setData(train_set,val_set,macroBatchSize=1)

    calcAndVis3DDescriptorsWithRot(descrNet,testdata_set,dataManager)
    plt.title('test data')
    
    ######################################
    # wait for quit
    
    cv2.waitKey(10)
    plt.show(block=True)
    
    ch = 0
    while ((ch & 0xFF) != ord('q')) and (ch >= 0):
        print("waiting for 'q'")
        ch = cv2.waitKey() 
        print(" ... got '{}' ({})".format(chr(ch & 0xFF),ch))
    
    cv2.destroyAllWindows()


        
    
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
    
    
    linemod_vis_test_descr_main(fileName,cfgparser)    
    
