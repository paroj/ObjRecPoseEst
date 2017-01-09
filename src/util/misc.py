'''
Created on Sep 2, 2014

@author: wohlhart
'''
import datetime, os
import cv2
import numpy
import theano
import colorsys
import copy

def nowStr():
    d = datetime.datetime.now()
    return d.strftime("%Y-%m-%d_%H-%M-%S")

def pidStr():
    return "pid{}".format(os.getpid())
    
def nowPidStr():
    return "{}-{}".format(nowStr(),pidStr())



def catPatchChannels(p1,p2):  # take two img/dpt patches  (h x w x m), (h x w x n) and make one (h x w x m+n)
    p1 = numpy.copy(p1)
    p2 = numpy.copy(p2)
    if p1.ndim == 2:  # grey level of depth ?
        p1 = p1.reshape((p1.shape[0],p1.shape[1],1))
    if p2.ndim == 2:  # grey level of depth ?
        p2 = p2.reshape((p2.shape[0],p2.shape[1],1))
        
    return numpy.concatenate((p1,p2),axis=2)
    
    
def addNoiseToTrainSeqs(trainSeqs,inputMode,rng,addSoftNoise=True,softNoiseRange=None,addHardNoise=True,addBgNoise=True):
    
    if softNoiseRange is None:
        W_bound = 0.1
    else:
        W_bound = softNoiseRange
    W_bound_rgb = 0.02 #W_bound #?
    print("W_bound {}".format(W_bound))       
    W_borders_low = 0.4
    W_borders_high = 0.8
    W_bound_fract = 0.5
    W_bound_fract_rgb = 0.2
    floatX = theano.config.floatX  # @UndefinedVariable
    
    for seqN in trainSeqs:
        seq = trainSeqs[seqN]
        for patch in seq.data:

            if inputMode != 1:
                # add noise to dpt
                
                #ramp = (1-numpy.tile(numpy.arange(patchSize).reshape(patchSize,1),(1,patchSize)) / float(patchSize)) - 0.5
                #print("mn/mx ramp {},{}".format(numpy.min(ramp),numpy.max(ramp)))
                #bg = 2*ramp + fractNoise
                #print("mn/mx bg {},{}".format(numpy.min(bg),numpy.max(bg)))
                
                #print("mn/mx {},{}".format(numpy.min(patch.dpt),numpy.max(patch.dpt)))

                # soft noise 
                if addSoftNoise:               
                    patch.dpt += numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=patch.dpt.shape),dtype=floatX)

                # hard noise on edges
                if addHardNoise:
                    hardNoise = numpy.asarray(rng.uniform(low=0., high=1., size=patch.dpt.shape),dtype=floatX) > 0.95
                    hardNoise = hardNoise.astype(floatX)
                    hardNoise = cv2.dilate(hardNoise,kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
                    edges = abs(cv2.Laplacian(patch.dpt,cv2.CV_32F))
                    edges = edges / numpy.max(edges)
                    hardNoiseEdges = hardNoise*edges
                    patch.dpt += numpy.asarray(rng.uniform(low=W_borders_low, high=W_borders_high, size=patch.dpt.shape),dtype=floatX) * hardNoiseEdges
                    patch.dpt = numpy.minimum(patch.dpt,1.)

                
                mask = patch.mask
                if addBgNoise and (mask is not None):
                    patchSize = patch.dpt.shape[0]
                    numLevels = numpy.ceil(numpy.log2(patchSize)).astype(numpy.int)
                    fractNoise = numpy.asarray(rng.uniform(low=-W_bound_fract, high=W_bound_fract, size=(2,2)),dtype=floatX)
                    #interp = cv2.INTER_NEAREST
                    interp = cv2.INTER_LINEAR
                    for i in range(numLevels-1):
                        #fractNoise1 = cv2.resize(fractNoise,(fractNoise.shape[0]*2,fractNoise.shape[1]*2),interpolation=cv2.INTER_NEAREST)
                        #fractNoise2 = cv2.resize(fractNoise,(fractNoise.shape[0]*2,fractNoise.shape[1]*2),interpolation=cv2.INTER_LINEAR)
                        #fractNoise = fractNoise1*0.6 + fractNoise2*0.4
                        fractNoise = cv2.resize(fractNoise,(fractNoise.shape[0]*2,fractNoise.shape[1]*2),interpolation=interp)
    
                        #b = W_bound_fract
                        b = W_bound_fract / numpy.sqrt(i+1)
                        #b = W_bound_fract / (i+1)
                        fractNoise += numpy.asarray(rng.uniform(low=-b, high=b, size=fractNoise.shape),dtype=floatX)
    
                        if i > 2:
                            fractNoise = cv2.medianBlur(fractNoise,5)
                        elif i > 1:
                            fractNoise = cv2.medianBlur(fractNoise,3)
                                                
                    if (fractNoise.shape[0] > patchSize) or (fractNoise.shape[1] > patchSize):
                        fractNoise = fractNoise[0:patchSize,0:patchSize]                
                    
                    patch.dpt += fractNoise * (1-mask) 


            if inputMode != 0:
                if addSoftNoise:               
                    patch.img += numpy.asarray(rng.uniform(low=-W_bound_rgb, high=W_bound_rgb, size=patch.img.shape),dtype=floatX)
                
                mask = copy.deepcopy(patch.mask)
                if addBgNoise and (mask is not None):
                    mask = numpy.tile(mask.reshape((mask.shape[0],mask.shape[1],1)),(1,1,3))
                    patchSize = patch.img.shape[0]
                    numLevels = numpy.ceil(numpy.log2(patchSize)).astype(numpy.int)
                    fractNoise = numpy.asarray(rng.uniform(low=-W_bound_fract_rgb, high=W_bound_fract_rgb, size=(2,2,3)),dtype=floatX)
                    #interp = cv2.INTER_NEAREST
                    interp = cv2.INTER_LINEAR
                    for i in range(numLevels-1):
                        #fractNoise1 = cv2.resize(fractNoise,(fractNoise.shape[0]*2,fractNoise.shape[1]*2),interpolation=cv2.INTER_NEAREST)
                        #fractNoise2 = cv2.resize(fractNoise,(fractNoise.shape[0]*2,fractNoise.shape[1]*2),interpolation=cv2.INTER_LINEAR)
                        #fractNoise = fractNoise1*0.6 + fractNoise2*0.4
                        fractNoise = cv2.resize(fractNoise,(fractNoise.shape[0]*2,fractNoise.shape[1]*2),interpolation=interp)
    
                        #b = W_bound_fract
                        b = W_bound_fract_rgb / numpy.sqrt(i+1)
                        #b = W_bound_fract / (i+1)
                        fractNoise += numpy.asarray(rng.uniform(low=-b, high=b, size=fractNoise.shape),dtype=floatX)
    
                        if i > 2:
                            fractNoise = cv2.medianBlur(fractNoise,5)
                        elif i > 1:
                            fractNoise = cv2.medianBlur(fractNoise,3)
                                                
                    if (fractNoise.shape[0] > patchSize) or (fractNoise.shape[1] > patchSize):
                        fractNoise = fractNoise[0:patchSize,0:patchSize]                
                    
                    patch.img += fractNoise * (1-mask) 
                    
                #cv2.imshow("noisy patch",patch.img+0.5)
                #cv2.waitKey()



def getNChanFromInputMode(inputMode):
    if inputMode == 0:   # DPT
        nChan = 1
    elif inputMode == 1: # IMG
        nChan = 3  
    else:                # DPT + IMG
        nChan = 4
        
    return nChan    
    

def getRGBFromHue(values):    
    h = values.flatten().reshape((len(values),1))
    s = numpy.ones((len(values),1))
    v = numpy.ones((len(values),1))
    hsv = numpy.hstack((h,s,v))    
    return numpy.array([colorsys.hsv_to_rgb(c[0],c[1],c[2]) for c in hsv])
    
    

# def readCfgParam(cfgparser,section,key,default):
#     if cfgparser.has_option(section,key):
#         return cfgparser.get(section,key)
#     else:
#         return default
#     
# def readCfgIntParam(cfgparser,section,key,default):    
#     if cfgparser.has_option(section,key):
#         return cfgparser.getint(section,key)
#     else:
#         return default
# 
# def readCfgFloatParam(cfgparser,section,key,default):    
#     if cfgparser.has_option(section,key):
#         return cfgparser.getfloat(section,key)
#     else:
#         return default
#     
# def readCfgBooleanParam(cfgparser,section,key,default):    
#     if cfgparser.has_option(section,key):
#         return cfgparser.getboolean(section,key)
#     else:
#         return default