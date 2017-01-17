'''
Created on Sep 25, 2014

@author: wohlhart
'''
import theano
import numpy
from util.misc import catPatchChannels
from data.camera import LinemodCam
import cv2
from data.transformations import dptInpaint
from tnetcore.data import BatchedDataset, BatchedTrainDataset
from itertools import izip

class BatchedImgSeqDataset(BatchedDataset):
    '''
    A BatchedDataset that can be initialized from an Image Sequence
    '''
    def __init__(self,x=None,y=None,yr=None,sampleInfo=None,batchSize=None):
        super(BatchedImgSeqDataset,self).__init__(x,y,yr,sampleInfo,batchSize)
        
    def initFromImgSeqs(self,imgSeqs,inputMode,batchSize):
        
        floatX = theano.config.floatX  # @UndefinedVariable
        
        #for seqN in imgSeqs:
        #    print("seqN {}".format(seqN))
            
        if isinstance(imgSeqs,dict):
            seqNames = sorted(imgSeqs.keys())  # fucking unsorted dicts, hate you
            imgSeqsList = []
            for i in range(len(seqNames)):
                imgSeqsList.append(imgSeqs[seqNames[i]])
            imgSeqs = imgSeqsList
        
        totalNumImgs = 0
        for i in xrange(len(imgSeqs)):
            totalNumImgs += len(imgSeqs[i].data)
                    
        imgD0 = dataArraysFromPatch(imgSeqs[0].data[0],inputMode)
        if inputMode == 3:
            print(imgD0[0].shape)
            print(imgD0[1].shape)
            print(len(imgD0))
            nChans = [imgD0[0].shape[0],imgD0[1].shape[0]]  # should be [3,1]
            patchH = imgD0[0].shape[1]
            patchW = imgD0[0].shape[2] 
            print("nChan {}, patchH {}, patchW {}".format(nChans,patchH,patchW))
        else:
            nChan,patchH,patchW = imgD0[0].shape
            nChans = [nChan]
            print("nChan, patchH, patchW {},{},{}".format(nChan,patchH,patchW))
        
        #nChan = getNChanFromInputMode(inputMode)
        numBatches = (totalNumImgs + batchSize - 1)//batchSize # integer ceil
        allData = [numpy.zeros((numBatches*batchSize,nChan,patchH,patchW),dtype=floatX) for nChan in nChans]
        allLabels = -numpy.ones((numBatches*batchSize,),dtype=numpy.int32)  # unused space in last minibatch must be filled with label=-1
        allRots = numpy.zeros((numBatches*batchSize,3),dtype=numpy.float32)
        allSampIdx = numpy.zeros((numBatches*batchSize,),dtype=numpy.int32)

        zRotInv = -numpy.ones((len(imgSeqs),),dtype=numpy.int32)
        
        k = 0    
        for i in xrange(len(imgSeqs)):
            for j in xrange(len(imgSeqs[i].data)):
                patch = imgSeqs[i].data[j]
                imgD = dataArraysFromPatch(patch,inputMode)
                for ad,imd in izip(allData,imgD):
                    ad[k] = imd
                allLabels[k] = i
                allRots[k] = patch.frame.pose.relCamPosZRotInv
                if zRotInv[i] < 0:
                    zRotInv[i] = patch.frame.pose.zRotInv
                elif zRotInv[i] != patch.frame.pose.zRotInv:
                    raise ValueError("zRotInv inconsistent, class {}: {} != {}".format(i,zRotInv[i],patch.frame.pose.zRotInv))
                allSampIdx[k] = j
                k += 1
        
        
        sampleInfo = {'maxSimTmplIdx': None, 
                      'sampIdx': allSampIdx,
                      'tmplBatchDataStartIdx': None,  
                      'rots': allRots, 
                      'tmplRots': None, 
                      'trainRots': None,
                      'zRotInv': zRotInv }
                        
        # store it
        if inputMode == 3:
            self.x = allData
        else:  
            self.x = allData[0]
        self.y = allLabels
        self.sampleInfo = sampleInfo
        self.batchSize = batchSize
        
        
class BatchedLinemodTrainDataset(BatchedTrainDataset):
    
    def __init__(self,x,y=None,yr=None,sampleInfo=None,pairIdx=None,pairLabels=None,tripletIdx=None,tripletPoolIdx=None,
                 nPairsPerBatch=None,nTripletsPerBatch=None,nTripletPoolsPerBatch=0,
                 negTripletPoolSizeVal=0, posTripletPoolSizeVal=0, batchSize=None):
        super(BatchedLinemodTrainDataset,self).__init__(x,y,yr,sampleInfo,pairIdx,pairLabels,tripletIdx,tripletPoolIdx,
                                                        nPairsPerBatch,nTripletsPerBatch,nTripletPoolsPerBatch,
                                                        negTripletPoolSizeVal,posTripletPoolSizeVal,batchSize)
        
    def append(self,other):
        """ concatenate the other BatchedTrainDataset to this one
        """
        super(BatchedLinemodTrainDataset,self).append(other)
                
        # TODO: check what the sampleInfo really contains. maybe make the sampleInfo a proper class?
        #sampleInfo = {'maxSimTmplIdx':allMaxSimTmplIdx, 'sampIdx':allSampIdx, 'tmplBatchDataStartIdx':allTmplBatchDataStartIdx,  'rots': allRots, 'tmplRots':tmplRots, 'trainRots':trainRots }
        self.sampleInfo['maxSimTmplIdx'] = numpy.concatenate((self.sampleInfo['maxSimTmplIdx'],other.sampleInfo['maxSimTmplIdx']))
        self.sampleInfo['sampIdx'] = numpy.concatenate((self.sampleInfo['sampIdx'],other.sampleInfo['sampIdx']))
        #self.sampleInfo['tmplBatchDataStartIdx'] = numpy.concatenate((self.sampleInfo['tmplBatchDataStartIdx'],other.sampleInfo['tmplBatchDataStartIdx']+nSamp))
        self.sampleInfo['tmplBatchDataStartIdx'] = numpy.concatenate((self.sampleInfo['tmplBatchDataStartIdx'],other.sampleInfo['tmplBatchDataStartIdx']))
        self.sampleInfo['rots'] = numpy.concatenate((self.sampleInfo['rots'],other.sampleInfo['rots']))
        
        ## not necessary. instead we should assert that they are the same, because the sampIdx is used to index them and its not adjusted 
        if not numpy.allclose(self.sampleInfo['tmplRots'], other.sampleInfo['tmplRots']):
            tr1 = self.sampleInfo['tmplRots']
            tr2 = other.sampleInfo['tmplRots']
            print(tr1.shape)
            print(tr2.shape)
            print("tr1 {}".format(tr1))
            print("tr2 {}".format(tr2))
            
        assert numpy.allclose(self.sampleInfo['trainRots'], other.sampleInfo['trainRots'])
        assert numpy.allclose(self.sampleInfo['zRotInv'], other.sampleInfo['zRotInv'])


    def extendSetBySubset(self,batchIdx):
        # batchIdx: which batches to replicate
        super(BatchedLinemodTrainDataset,self).extendSetBySubset(batchIdx)
        
        sampIdxInBatch = numpy.tile(numpy.arange(self.batchSize),len(batchIdx))
        batchOffset = numpy.repeat(batchIdx,self.batchSize)*self.batchSize
        sampIdx = batchOffset + sampIdxInBatch
        
        self.sampleInfo['maxSimTmplIdx'] = numpy.concatenate((self.sampleInfo['maxSimTmplIdx'],self.sampleInfo['maxSimTmplIdx'][sampIdx]))
        self.sampleInfo['sampIdx'] = numpy.concatenate((self.sampleInfo['sampIdx'],self.sampleInfo['sampIdx'][sampIdx]))
        self.sampleInfo['tmplBatchDataStartIdx'] = numpy.concatenate((self.sampleInfo['tmplBatchDataStartIdx'],self.sampleInfo['tmplBatchDataStartIdx'][batchIdx]))
        self.sampleInfo['rots'] = numpy.concatenate((self.sampleInfo['rots'],self.sampleInfo['rots'][sampIdx]))
        

class LinemodBatchedTestData(BatchedImgSeqDataset):
    
    def __init__(self,x=None,y=None,yr=None,sampleInfo=None,batchSize=None):
        super(LinemodBatchedTestData,self).__init__(x,y,yr,sampleInfo,batchSize)

    def addFromCroppedGTpatches(self,lmDataset,inputMode=0,targetSize=64,cropSize=20.0,batchSize=None):
        '''
        cropPatches from Linemod test data frames at GT position
        and assemble in x
        '''
        floatX = theano.config.floatX  # @UndefinedVariable
        
        totalNumImgs = 0
        for seq in lmDataset.imgSeqs:
            totalNumImgs += len(seq.data)
            
        seq0 = lmDataset.imgSeqs[0]
        frame0 = seq0.data[0]
        if inputMode == 0:   # DPT
            nChans = [1]
        elif inputMode == 1: # IMG
            nChans = [frame0.img.shape[2]] # ie 3
        elif inputMode == 2:                # DPT + IMG
            nChans = [4] # or [frame0.img.shape[2]+1]
        else:
            nChans = [3, 1]
            
        # TODO: input mode == 2   IMG / DPT
        
        patchH = targetSize
        patchW = targetSize
        
        print("totalNumImgs {}".format(totalNumImgs))
        
        if batchSize is None:
            batchSize = 100
        
        assert (self.batchSize is None) or (self.batchSize == batchSize), "self.batchSize {} != new batchSize {}".format(self.batchSize,batchSize) 
        
        numBatches = numpy.ceil(float(totalNumImgs)/float(batchSize))
        allData = [numpy.zeros((numBatches*batchSize,nChan,patchH,patchW),dtype=floatX) for nChan in nChans]
        allLabels = -numpy.ones((numBatches*batchSize,),dtype=numpy.int32)  # unused space in last minibatch must be filled with label=-1
        allRots = numpy.zeros((numBatches*batchSize,3),dtype=numpy.float32)

        minClass = 0
        if self.y is not None:
            minClass = numpy.max(self.y)+1
            
        k = 0
        seqN = 0
        for seq in lmDataset.imgSeqs:
            for i in xrange(len(seq.data)):
                frame = seq.data[i]
                imgD = self.cropPatchAtGTPos(frame,cropSize,targetSize,inputMode)
                    
                #print(imgD.shape)
                for allChannelData,channelData in izip(allData,imgD):
                    allChannelData[k] = channelData
                allLabels[k] = seqN + minClass
                allRots[k] = frame.pose.relCamPosZRotInv
                
                k += 1
            
            seqN += 1
            
        if self.x is None:
            print("#-# set")
            self.x = allData
            self.y = allLabels
            self.sampleInfo = {'rots':allRots}
            self.batchSize = batchSize
        else:
            print("#-# add")
            self.x = [numpy.concatenate((x,d)) for x,d in izip(self.x,allData)] if (inputMode == 3) else numpy.concatenate((self.x,allData[0]))
            self.y = numpy.concatenate((self.y,allLabels))
            self.sampleInfo['rots'] = numpy.concatenate((self.sampleInfo['rots'],allRots))
            #self.batchSize = batchSize
            
        print("num samples: {}".format(self.numSamples))
        #sampleInfo = {'maxSimTmplIdx':allMaxSimTmplIdx, 'sampIdx':allSampIdx }
    
    
    def cropPatchAtGTPos(self,frame,cropSize,targetSize,inputMode):
        
        lcam = LinemodCam()
        
        mat = frame.pose.getMatrix()
        worldPt = numpy.array([0,0,-5,1.])
        camPt = numpy.dot(mat,worldPt)  # center pt in camera space
        screenPt = lcam.worldToScreen(camPt)

        v = camPt[0:3]
        
        ignoreInplaneRotation = False
        if ignoreInplaneRotation:
            # up vector of rectangle to crop in the world = x-axis X v = (1,0,0) X (vx,vy,vz) = (0,-vz,vy)    (or really it is down (depending on the coord system))
            u = numpy.array([0.,-v[2],v[1]])
            # right vector of rectangle to crop = u X v = (wy^2+cropDepth^2,-wx*wy,-wx*cropDepth) = (vy^2+vz^2, vx*vy, -vy*vz) 
            r = numpy.array([v[1]*v[1]+v[2]*v[2],-v[0]*v[1],-v[0]*v[2]])
        else:
            upV = (numpy.dot(mat,numpy.array([0,0,-6,1.])) - camPt)[0:3]
            #rightV = (numpy.dot(mat,numpy.array([1,0,-5,1.])) - camPt)[0:3]
            #print()
            
            r = numpy.cross(v,upV)
            u = numpy.cross(r,v)
            #u = numpy.cross(rightV,v)
            #r = numpy.cross(u,v)
            
        # normalize, resize
        u = u * (cropSize/2) / numpy.linalg.norm(u)
        r = r * (cropSize/2) / numpy.linalg.norm(r)
        
        # world coord of corner points
        w1 = v + u - r
        w2 = v - u - r
        w3 = v + u + r
        w4 = v - u + r
        
        # coords of corner points on image plane
        worldRect = numpy.concatenate((w1.reshape((1,3)),w2.reshape((1,3)),w3.reshape((1,3)),w4.reshape((1,3))),axis=0)
        #print("worldRect.shape {}".format(worldRect.shape))
        sourceRect = lcam.worldToScreen(worldRect).astype(numpy.float32)
        #print("sourceRect.shape {}".format(sourceRect.shape))
        #print("sourceRect {}".format(sourceRect))
        #print("type(sourceRect[0,0]) {}".format(type(sourceRect[0,0])))
              
        # target pts 
        targetRect = numpy.array([[0,0],[0,1],[1,0],[1,1]]).astype(numpy.float32) * targetSize 
        M = cv2.getPerspectiveTransform(sourceRect,targetRect)
        
        # DEBUG
#         print("{} -> {}".format(sourceRect,targetRect))
#         im = numpy.copy(frame.img)
#         im += 0.5
#         cv2.line(im,tuple(sourceRect[0,:]),tuple(sourceRect[1,:]),color=[0,0,255])
#         cv2.line(im,tuple(sourceRect[1,:]),tuple(sourceRect[2,:]),color=[0,0,255])
#         cv2.line(im,tuple(sourceRect[2,:]),tuple(sourceRect[3,:]),color=[0,0,255])
#         cv2.line(im,tuple(sourceRect[3,:]),tuple(sourceRect[0,:]),color=[0,0,255])
#         cv2.imshow("test",im)
        
        ts = (targetSize,targetSize)
        if inputMode != 1:
            # frame.dpt is data in float32 measuring depth in cm
            dpt = numpy.copy(frame.dpt)
            
            # fix holes
            fixHolesMode = -1  
            if fixHolesMode == 0:
                dpt[dpt==0] = numpy.max(dpt)+1.
            elif fixHolesMode == 1:
                inpaintMask = (dpt == 0).astype(numpy.uint8)
                #cv2.imshow("mask",inpaintMask*255)
                #cv2.imshow("src dpt",dpt/400.)

                #dpt = cv2.inpaint(dpt,inpaintMask,inpaintRadius=3,flags=cv2.INPAINT_TELEA)  # can only be used on uint8
                 
                dpt = dptInpaint(dpt,inpaintMask)
                
                #cv2.imshow("dpt",dpt/400.)
                #cv2.waitKey()
            
            #print("center point dst {}".format(frame.dpt[screenPt[1],screenPt[0]]))
            #print("type: {}".format(type(dpt[0][0])))  
            #print("{},{}".format(numpy.min(dpt),numpy.max(dpt)))
            
            dptPatch = cv2.warpPerspective(dpt,M,ts)
            
            dHigh = 20. 
            dLow = -20.
            dptPatch = dptPatch - camPt[2]  # subtract depth of extraction point (center values on it)
            dptPatch = numpy.maximum(numpy.minimum(dptPatch,dHigh),dLow)
            dptPatch /= numpy.maximum(abs(dHigh),abs(dLow)) # -> / 20.  => -1..1
            
            imgD = dptPatch
#            print("{},{}".format(numpy.min(imgD),numpy.max(imgD)))
            #cv2.imshow("cropped",imgD)
            
        if inputMode != 0:
            imgPatch = cv2.warpPerspective(frame.img+0.5,M,ts)-0.5  # do we need this +/- 0.5, or can the values be negative in the process? 
            
            
        if inputMode == 0:
            imgD = dptPatch 
        elif inputMode == 1:
            imgD = imgPatch             
            imgD = numpy.rollaxis( imgD , 2)
        else:
            imgD = catPatchChannels(imgPatch,dptPatch)
            imgD = numpy.rollaxis( imgD , 2)
            
        #cv2.waitKey(0)
            
        return imgD


def dataArraysFromPatch(patch,inputMode):
    
    imgD = None
    
    if inputMode == 0:
        
        if patch.dpt is not None:
            imgD = [patch.dpt.reshape((1,patch.dpt.shape[0],patch.dpt.shape[1]))]
        else:
            print("WARNING: batcheddatasets.dataArraysFromPatch:  patch.dpt is empty")                                      

    elif inputMode == 1:
        
        if patch.img is not None:
            imgD = patch.img
            imgD = [numpy.rollaxis( imgD , 2)]
        else:
            print("WARNING: batcheddatasets.dataArraysFromPatch:  patch.img is empty")
            
    elif inputMode == 2:
#         imgDdpt = numpy.copy(patch.dpt)
#         imgDdpt = imgD.reshape((1,imgDdpt.shape[0],imgDdpt.shape[1]))
#         imgDimg = numpy.copy(patch.img)
#         imgDimg = numpy.rollaxis(imgDimg,2)
#         imgD = numpy.concatenate((imgDimg,imgDdpt),axis=0)
        
        if (patch.img is not None) and (patch.dpt is not None):
            # v2
            imgD = catPatchChannels(patch.img,patch.dpt)
            imgD = [numpy.rollaxis(imgD,2)]
        else:
            print("WARNING: batcheddatasets.dataArraysFromPatch:  patch.img or patch.dpt is empty")
            
    elif inputMode == 3:
        
        if (patch.img is not None) and (patch.dpt is not None):
            imgD = catPatchChannels(patch.img,patch.dpt)
            imgD = numpy.rollaxis(imgD,2)
            imgD = [imgD[0:3], imgD[3].reshape((1,imgD.shape[1],imgD.shape[2]))]
        else:
            print("WARNING: batcheddatasets.dataArraysFromPatch:  patch.{} is empty".format("img" if patch.img is None else "dpt"))        
            
    else:
        raise ValueError("Unknown input mode: {}".format(inputMode)) 
        
    return imgD


