'''
Created on Jul 18, 2014

@author: wohlhart
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # @UnresolvedImport @UnusedImport

import numpy, cv2
from PIL import Image
import scipy.spatial.distance
import colorsys
import os
import theano

from data.camera import LinemodCam
from tnetcore.layers.hiddenlayer import HiddenLayer
from tnetcore.layers.convlayer import ConvLayer
from IPython.kernel.zmq.serialize import cPickle
from util.matplotlib2tikz import save as tikz_save
from util.misc import getRGBFromHue
from tnetcore.layers.convpoollayer import ConvPoolLayer
    
def showDistanceMatrix(dst):
    
    imgD = ((dst - numpy.min(dst))*255.0/(numpy.max(dst) - numpy.min(dst))).astype(numpy.uint8)
    #imgD = imgD.reshape((imgD.shape[0],imgD.shape[1],1)).repeat(3, axis=2)
    #img = Image.fromarray(imgD,mode='RGB') 
    img = Image.fromarray(imgD,mode='L')
    
#     fig = plt.figure(tight_layout=True)
#     ax = fig.add_subplot(1,1,1)
#     ax.axes.set_frame_on(False)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     
#     p = ax.imshow(img)
#     p.set_interpolation('nearest')
#     fig.show()
#     plt.show(block=False)        
    showMat(img)


def showMat(img):    
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(1,1,1)
    ax.axes.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
        
    p = ax.imshow(img)
    p.set_interpolation('nearest')
    fig.show()
    plt.show(block=False)        

    
def checkFiltersDist(descrNet):
    wvals = descrNet.layer0.W.get_value()
    wvals = wvals.reshape((wvals.shape[0],numpy.prod(wvals.shape[1:])))
    dst = scipy.spatial.distance.pdist(wvals,'cosine')
    dst = scipy.spatial.distance.squareform(dst)
    showDistanceMatrix(dst)    
    
    
def visualizeFiltersEvolution(wvals,fileName=None):
    ####################################################
    #
    #  visualize filters
    #
    #wval = network.descr_net0.layer0.W.get_value()
    #wvals.append(network.descr_net0.layer0.W.get_value())
    
    numInterm = min(len(wvals),10)
    idx = numpy.linspace(0,len(wvals)-1,num=numInterm).round().astype(int)
    
    fig = plt.figure(tight_layout=True);
    for j in range(len(idx)):
        wval = wvals[idx[j]]
        N = wval.shape[0]
        for i in range(N):
            ax = fig.add_subplot(numInterm,N,j*N+i+1)
            ax.axes.set_frame_on(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
                    
            if wval[i].shape[0] > 1:
                imgD = numpy.swapaxes(numpy.swapaxes(wval[i],0,1),1,2)
            else:
                imgD = wval[i].squeeze()
                
            imgD = ((imgD - numpy.min(imgD))*255.0/(numpy.max(imgD) - numpy.min(imgD))).astype(numpy.uint8)
            if len(imgD.shape) > 2:
                img = Image.fromarray(imgD,mode='RGB') 
            else: 
                #img = Image.fromarray(imgD) #,mode='L')  # this is color coded
                imgD = imgD.reshape((imgD.shape[0],imgD.shape[1],1)).repeat(3, axis=2)
                img = Image.fromarray(imgD,mode='RGB') #,mode='L') 
                    
            p = ax.imshow(img)
            p.set_interpolation('nearest')

    if fileName == None:
        fig.show()
        plt.show(block=False)
    else:
        plt.savefig(fileName)        
    

def visualizeNetworkFilters(network,fileName=None):
    ####################################################
    #
    #  visualize filters of first two convolution layers
    #
    #    very hacky, many assumptions about the network architecture 
    #
    
    if not fileName is None:
        fn,ext = os.path.splitext(fileName)
        fn0 = fn + '_layer0' + ext
        fn1 = fn + '_layer1' + ext
    else:
        fn0 = None
        fn1 = None
    
    ####### get conv filters from first to convlayers
    wvals0 = None
    wvals1 = None
    for l in network.layers:
        if isinstance(l,ConvLayer) or isinstance(l,ConvPoolLayer):
            if wvals0 is None:
                wvals0 = l.W.get_value()
            else:
                wvals1 = l.W.get_value()
                break   

    Wout = numpy.sum(abs(wvals1),axis=3)
    Wout = numpy.sum(Wout,axis=2)
    Wout = numpy.max(Wout,axis=0)
    
    Wout = Wout / numpy.max(Wout)
    
    visFilters(wvals0,Wout,fileName=fn0)

    ####### layer 1
    
#     # find layer with weights after layer1
#     i = 0
#     for l in network.layers:
#         if isinstance(l,HiddenLayer) or isinstance(l,ConvLayer) or isinstance(l,ConvPoolLayer):
#             if i < 2:
#                 i = i + 1
#             else:
#                 nextLayer = l
#                 break
#     
#     wvals0 = wvals1
#     nFilters1 = wvals1.shape[0]  # nFilters, nIn, h, w
#     nIn1 = wvals1.shape[1]
#     w1 = wvals1.shape[2]
#     h1 = wvals1.shape[3]
#     W2 = nextLayer.W.get_value()   # shape should be  nF1*hout1*wout1, layer2_outputdim
#     if isinstance(nextLayer, HiddenLayer):
#         n2in = W2.shape[0]
#         W2 = numpy.sum(abs(W2),axis=1) #  sum over layer2 output dim
#         W2 = W2.reshape((nFilters1,n2in/nFilters1))          # -> one row per layer1 filter 
#         Wout = numpy.sum(W2,axis=1)
#     else:
#         Wout = numpy.sum(W2,axis=3)
#         Wout = numpy.sum(Wout,axis=2)
#         Wout = numpy.sum(Wout,axis=0)
#     #Wout = Wout.reshape((nFilters1,1))
#     #Wout = numpy.tile(Wout,(1,nIn1)).reshape((nFilters1*nIn1,)) # replicate
#     #wvals0 = wvals0.reshape((nFilters1*nIn1,1,h1,w1))
#     wvals0 = numpy.swapaxes(wvals0,1,2)  # nFilters,nIn1,h,w -> nFilters,h,nIn1,w
#     wvals0 = numpy.reshape(wvals0,(nFilters1,1,h1,nIn1*w1))  
#     visFilters(wvals0,Wout,nFilters1,fileName=fn1)
    
def visFilterMontageOfLayer(wvals,fileName=None,windowName=None):
    
    numFilters,nChan,fh,fw = wvals.shape  

    #if nChan == 3:
    #    nChanOut = 3
    #else:
    #    nChanOut = 1

    # if we have more than 3 channels, montage the next to each other
    #if nChan > 3:
    #    targetFW = fw*nChan
    #else:
    #    targetFW = fw

    border = 1
    nChanOut = 1
    targetFW = fw*nChan
    targetFH = fh
    
    filterAspect = float(targetFW + border)/float(targetFH+border)
    targetAspect = 4.0/3.0

#     print("targetFW = {}".format(targetFW))
#     print("targetFH = {}".format(targetFH))
#     print("filterAspect = {}".format(filterAspect))
#     print("targetAspect = {}".format(targetAspect))
#     print("numFilters = {}".format(numFilters))

    numCols = numpy.floor(numpy.sqrt(numFilters * targetAspect/filterAspect))
    numCols = numpy.maximum(numCols,1)
    numRows = numpy.ceil(numFilters/float(numCols))
    
#     print("numCols = {}".format(numCols))
#     print("numRows = {}".format(numRows))
    
    rowHeight = targetFH+border
    colWidth = targetFW+border
            
    wAll = numpy.zeros((numRows*rowHeight,numCols*colWidth),dtype=numpy.float32)
    wFrame = numpy.zeros((numRows*rowHeight,numCols*colWidth),dtype=numpy.float32)
    for i in xrange(numFilters):
        
        colIdx = i % numCols
        rowIdx = i // numCols
        
        filterPatch = wvals[i]
        if nChan > 1:
            filterPatch = numpy.swapaxes(filterPatch,0,1).reshape(fh,fw*nChan)  # all channels next to each other
        
        wAll[rowIdx*rowHeight:rowIdx*rowHeight+targetFH,colIdx*colWidth:colIdx*colWidth+targetFW] = filterPatch
        wFrame[rowIdx*rowHeight:rowIdx*rowHeight+targetFH,colIdx*colWidth:colIdx*colWidth+targetFW] = numpy.ones(filterPatch.shape)
        
    wAllMin = numpy.min(wAll)
    wAllMax = numpy.max(wAll)
    wAll = (wAll-wAllMin)/(wAllMax-wAllMin)
    wAll = wAll*wFrame
    
    if fileName is None:
        cv2.imshow(windowName,wAll)
    else:
        cv2.imwrite(fileName,wAll*255.0)
    
    
def visNetworkFiltersMontage(network,fileName=None):
    ####################################################
    #
    #  visualize and save filters of first two convolution layers
    #
    #    very hacky, many assumptions about the network architecture 
    #
    
    ####### get conv filters from first to convlayers
    i = 0
    windowName = None
    layerFN = None
    for l in network.layers:
        if isinstance(l,ConvLayer) or isinstance(l,ConvPoolLayer):
            wvals = l.W.get_value()

            if fileName is None:
                windowName = 'layer{}'.format(i)                
            else:
                fn,ext = os.path.splitext(fileName)
                layerFN = fn + '_layer{}'.format(i) + ext

            visFilterMontageOfLayer(wvals,fileName=layerFN,windowName=windowName)

        i = i+1
        #break
   

    
def visFilters(wvals,Wout,numRows=None,fileName=None):
    
    numFilters = wvals.shape[0]
    if numRows is None:
        numRows = numpy.floor(numpy.sqrt(numFilters))
    numCols = numpy.ceil(numFilters/float(numRows))
    
    #wvals = ((wvals - numpy.min(wvals))*255.0/(numpy.max(wvals) - numpy.min(wvals))).astype(numpy.uint8)
    #wmn = numpy.min(wvals)
    #wmx = numpy.max(wvals)
    #wvals = (wvals - wmn)/(wmx - wmn)
    
    nChan = wvals[0].shape[0]
    
    fig = plt.figure(tight_layout=True)
    
    for i in range(numFilters):
        ax = fig.add_subplot(numRows,numCols,i+1)
        ax.axes.set_frame_on(False)
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        ax.grid(False)
                
        if nChan > 1:
            imgD = numpy.swapaxes(numpy.swapaxes(wvals[i],0,1),1,2)
            if nChan == 4:  
                imgD = numpy.hstack((imgD[:,:,:3],numpy.tile(imgD[:,:,3].reshape(imgD.shape[0],imgD.shape[1],1),(1,1,3))))                
        else:
            imgD = wvals[i].squeeze()
            
        #imgD = ((imgD - numpy.min(imgD))*255.0/(numpy.max(imgD) - numpy.min(imgD))).astype(numpy.uint8)
        wmn = numpy.min(imgD)
        wmx = numpy.max(imgD)
        imgD = (imgD - wmn)/(wmx - wmn)


        if len(imgD.shape) > 2:            
            #img = Image.fromarray(imgD,mode='RGB') 
            p = ax.imshow(imgD)
            p.set_interpolation('nearest')
        else: 
            ax.matshow(imgD,cmap=plt.cm.gray)       # @UndefinedVariable
            ##img = Image.fromarray(imgD) #,mode='L')  # this is color coded
            #imgD = imgD.reshape((imgD.shape[0],imgD.shape[1],1)).repeat(3, axis=2)
            #img = Image.fromarray(imgD,mode='RGB') #,mode='L') 
        
        #wstr = "{:.2f}".format(Wout[i])
        #wstr = "{:.3e}".format(Wout[i])
        wstr = "{:.3f},{:.3f}. o {:.3f}".format(wmn,wmx,Wout[i])
        ax.set_xlabel(wstr)

    if fileName == None:
        fig.show()
        plt.show(block=False)
    else:
        plt.savefig(fileName)        
    
        
    ##### vis layer1 filters
    # TODO
    
    
def visualizeNetworkResponse(descrNet,imgStack):
    '''
    show the response maps of the layers in the network to a given input image
    '''
    
    if imgStack.ndim == 3:  # only one image. chan x h x w
        imgStack = imgStack.reshape((1,imgStack.shape[0],imgStack.shape[1],imgStack.shape[2]))
        
    numImgs = imgStack.shape[0]
    
    outputs = [layer.output for layer in descrNet.layers if isinstance(layer,ConvLayer)]
    f = theano.function(inputs=[],outputs=outputs,givens={descrNet.inputVar:imgStack})
    layerOutputs = f()
    
    #with open('tmp.pkl','wb') as f:
    #    cPickle.dump(layerOutputs,f,protocol=cPickle.HIGHEST_PROTOCOL)
    
    for i in range(numImgs):
        
        # show input image
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axes.set_frame_on(False)
        ax.grid(False)
        imgD = numpy.copy(imgStack[i])
        if imgD.shape[0] > 1:
            imgD = numpy.swapaxes(numpy.swapaxes(imgD,0,1),1,2)
        else:
            imgD = imgD.squeeze()

#        cv2.imshow("fuck",imgD+0.5)
#        cv2.waitKey(0)
            
        if len(imgD.shape) > 2:
            imgD = numpy.round((imgD + 0.5)*255.)
            img = Image.fromarray(imgD.astype(numpy.uint8),mode='RGB') 
            p = ax.imshow(img)
            p.set_interpolation('nearest')
        else: 
            ax.matshow(imgD,cmap=plt.cm.gray)       # @UndefinedVariable
            
        for j in range(len(descrNet.layers)):
            if isinstance(descrNet.layers[j],ConvLayer):
                # show response maps in one big map        
                resp = layerOutputs[j][i]
                respMaps = montageMaps(resp)
                respMaps = (respMaps - numpy.min(respMaps))/(numpy.max(respMaps) - numpy.min(respMaps)) 
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.axes.set_frame_on(False)
                ax.grid(False)
                ax.matshow(respMaps,cmap=plt.cm.gray)       # @UndefinedVariable
    
#     layerF = []
#     for j in range(len(descrNet.layers)):
#         layer = descrNet.layers[j]
#         if isinstance(layer,ConvLayer):
#             f = theano.function(inputs=[],outputs=[layer.output],givens={descrNet.inputVar:imgStack})
#             layerF.append(f)
#         else:
#             layerF.append(None)
#     
#     for i in range(numImgs):
#         
#         # show input image
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.axes.set_frame_on(False)
#         ax.grid(False)
#         imgD = imgStack[i]
#         if imgD.shape[0] > 1:
#             imgD = numpy.swapaxes(numpy.swapaxes(imgD,0,1),1,2)
#         else:
#             imgD = imgD.squeeze()
#             
#         if len(imgD.shape) > 2:
#             img = Image.fromarray(imgD,mode='RGB') 
#             p = ax.imshow(img)
#             p.set_interpolation('nearest')
#         else: 
#             ax.matshow(imgD,cmap=plt.cm.gray)       # @UndefinedVariable
#             
#         for j in range(len(descrNet.layers)):
#             if isinstance(descrNet.layers[j],ConvLayer):
#                 layerOutput = layerF[j]()
#                 layerOutput = layerOutput[0]
#         
#                 # show response maps in one big map        
#                 resp = layerOutput[i]
#                 respMaps = montageMaps(resp)
#                 respMaps = (respMaps - numpy.min(respMaps))/(numpy.max(respMaps) - numpy.min(respMaps)) 
#                 fig = plt.figure()
#                 ax = fig.add_subplot(111)
#                 ax.axes.set_frame_on(False)
#                 ax.grid(False)
#                 ax.matshow(respMaps,cmap=plt.cm.gray)       # @UndefinedVariable
        
           
    

def montageMaps(maps,numRows=None):
    '''
    maps = chan x h x w
    output: montage of all maps next to each other, channels separated, organized in a square (or in given number of cols)
    '''
    nMaps = maps.shape[0]
    print maps.shape
    if numRows is None:
        numRows = numpy.floor(numpy.sqrt(nMaps))
    numCols = numpy.ceil(nMaps/float(numRows))
    h = maps.shape[1] 
    w = maps.shape[2]
    resMaps = numpy.zeros((h*numRows,w*numCols),dtype=numpy.float32)
    for j in range(nMaps):
        x = numpy.mod(j,numCols)  
        y = numpy.floor(j / numCols)
        
        #print x,y,w,h
        m = numpy.copy(maps[j])
        # normalize individually ?
        if numpy.max(m) > numpy.min(m):
            m = (m - numpy.min(m))/(numpy.max(m)-numpy.min(m))
        resMaps[y*h:(y+1)*h,x*w:(x+1)*w] = m
    
    return resMaps    
    
    
def montageLinemodSamples(seq):

    maxFigureWidth = 1800
    maxFigureHeight = 1800
    
    nImgs = len(seq.data)
    f0 = seq.data[0]
    h = f0.img.shape[0]
    w = f0.img.shape[1]
    nCols = maxFigureWidth / w
    nRows = min(maxFigureHeight/h,nImgs/nCols+1) 
    imgs = numpy.zeros((nRows*h,nCols*w,3),f0.img.dtype)
    dpts = numpy.zeros((nRows*h,nCols*w),f0.dpt.dtype)
    n = 0
    for i in xrange(min(len(seq.data),nCols*nRows)):
        frame = seq.data[i]
        x = n % nCols
        y = n / nCols 
        imgs[y*h:(y+1)*h,x*w:(x+1)*w,:] = frame.img
        dpts[y*h:(y+1)*h,x*w:(x+1)*w] = frame.dpt
        n = n+1

    cv2.imshow("{}_imgs".format(seq.name),imgs)
    cv2.imshow("{}_dpts".format(seq.name),dpts/8.)
    #cv2.waitKey(0)    
    
def montageLinemodTrainSeqSamples(trainSeq,inputMode=0):
    maxFigureWidth = 1800
    maxFigureHeight = 1800
    
    nImgs = len(trainSeq.data)
    f0 = trainSeq.data[0].dpt
    h = f0.shape[0]
    w = f0.shape[1]
    nCols = maxFigureWidth / w
    nRows = min(maxFigureHeight/h,nImgs/nCols+1) 
    #imgs = numpy.zeros((nRows*h,nCols*w,3),f0.img.dtype)
    dpts = numpy.zeros((nRows*h,nCols*w),f0.dtype)
    n = 0
    for i in xrange(min(nImgs,nCols*nRows)):
        x = n % nCols
        y = n / nCols 
        #imgs[y*h:(y+1)*h,x*w:(x+1)*w,:] = frame.img
        dpts[y*h:(y+1)*h,x*w:(x+1)*w] = trainSeq.data[i].dpt
        n = n+1
    
    print("mx {}, mn {}".format(numpy.max(dpts),numpy.min(dpts)))

    #cv2.imshow("{}_imgs".format(seq.name),imgs)
    #cv2.imshow("dpts",dpts/8.)
    cv2.imshow("dpts",(dpts+1.)/2.)
    #cv2.waitKey(0)            
    
      
def montageLinemodTrainSetSample(train_set):
    
    maxFigureWidth = 1800
    maxFigureHeight = 1800
    
    nImgs = train_set.numSamples
    if isinstance(train_set.x,list):
        f0 = numpy.concatenate([x[0] for x in train_set.x],axis=0)
    else:
        f0 = train_set.x[0]
        
    h = f0.shape[1]
    w = f0.shape[2]
    nCols = maxFigureWidth / w
    nRows = min(maxFigureHeight/h,nImgs/nCols+1) 
    #imgs = numpy.zeros((nRows*h,nCols*w,3),f0.img.dtype)
    dpts = numpy.zeros((nRows*h,nCols*w),f0.dtype)
    n = 0
    for i in xrange(min(nImgs,nCols*nRows)):
        x = n % nCols
        y = n / nCols 
        #imgs[y*h:(y+1)*h,x*w:(x+1)*w,:] = frame.img
        if isinstance(train_set.x,list):
            fts = numpy.concatenate([x[i] for x in train_set.x],axis=0)
        else:
            fts = train_set.x[i]
        dpts[y*h:(y+1)*h,x*w:(x+1)*w] = fts.reshape((h,w))
        n = n+1
    
    print("mx {}, mn {}".format(numpy.max(dpts),numpy.min(dpts)))

    #cv2.imshow("{}_imgs".format(seq.name),imgs)
    #cv2.imshow("dpts",dpts/8.)
    cv2.imshow("dpts",dpts+0.5)
    #cv2.waitKey(0)            
    
    
def montageLinemodTrainSamplesAndTmpls(train_set,outFileName=None):
    
    maxFigureWidth = 1800
    maxFigureHeight = 1800
    
    border = 2
    nImgs = train_set.numSamples
    if isinstance(train_set.x,list):
        f0 = numpy.concatenate([x[0] for x in train_set.x],axis=0)
    else:
        f0 = train_set.x[0]
    nChan,h,w = f0.shape
    if nChan > 3:
        w = 2*w  # rgb + depth, next to each other
        nChan = 3
    nCols = maxFigureWidth / ((w*2)+border)
    nRows = min(maxFigureHeight/h,nImgs/nCols+1) 
    imgs = numpy.zeros((nRows*h,nCols*(w*2 + border),nChan),f0.dtype)
    n = 0
    for i in xrange(min(nImgs,nCols*nRows)):

        #skip tmpls
        batchOffset = (numpy.floor(i / train_set.batchSize)*train_set.batchSize).astype(numpy.int) 
        trainIdx = train_set.tripletIdx[i][0] + batchOffset 
        tmplIdx = train_set.tripletIdx[i][1] + batchOffset 
        
        if isinstance(train_set.x,list):
            trs = numpy.concatenate([x[trainIdx] for x in train_set.x],axis=0)
            ts = numpy.concatenate([x[tmplIdx] for x in train_set.x],axis=0)
        else:
            trs = train_set.x[trainIdx]
            ts = train_set.x[tmplIdx]
        trainSample = sampleToPatch(trs)
        tmpl = sampleToPatch(ts)
        
        x = i % nCols
        y = i / nCols 

        l = x*(w*2 + border)
        t = y*h
        #print("trainSample.shape {}; {},{}, {},{}".format(trainSample.shape,h,w,t,l))
        imgs[t:t+h,l:l+w] = trainSample
        imgs[t:t+h,l+w:l+2*w] = tmpl
        
        print("{}: {},{}".format(i,train_set.sampleInfo['rots'][trainIdx],train_set.sampleInfo['rots'][tmplIdx]))
    
    mx = numpy.max(imgs)
    mn = numpy.min(imgs)
    print("mx {}, mn {}".format(mx,mn))
    
    imgs = numpy.clip(imgs,0.0,1.0)

    mx = numpy.max(imgs)
    mn = numpy.min(imgs)
    print("mx {}, mn {}".format(mx,mn))
    
    #cv2.imshow("{}_imgs".format(seq.name),imgs)
    #cv2.imshow("dpts",dpts/8.)
    #cv2.imshow("dpts",(imgs+1.)/2.)
    cv2.imshow("samples",imgs)
    #cv2.waitKey(0)            
    
    if outFileName is not None:
        cv2.imwrite(outFileName,imgs*255.)
    
def calcAndVis3DDescriptors(network,test_data,fileName=None):

    res = network.computeDescriptors(test_data.x)
    vis3DDescriptors2(res,test_data.y,fileName)
        
def vis3DDescriptors2(descrs,labels,fileName=None):

    #cIdx = test_data.y == 0
    #sIdx = test_data.y == 1
    #tIdx = test_data.y == 2  
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    markers = ['o','s','^']
    colors = ['b','r','g']
    for i in range(3):
        idx = labels == i
        ax.scatter3D(descrs[idx,0],descrs[idx,1],descrs[idx,2],c=colors[i], marker=markers[i])
    
    if fileName==None:
        plt.show(block=False)
    else:
        plt.savefig(fileName)
        

def calcAndVis3DDescriptorsWithRotWithTrainer(trainer,test_data,fileName=None):

    print("numSamples {}, batch_size {}".format(test_data.numSamples,test_data.batchSize))

    res = trainer.computeDescriptors(test_data)
    # throw away results for invalid samples (that were added to fill up minibatches)
    validIdx = test_data.y >= 0
    print("test_data.x.shape {}, num valid {}".format(test_data.numSamples,numpy.sum(validIdx)))
    vis3DDescriptorsWithRot(res[validIdx],test_data.y[validIdx],test_data.sampleInfo['rots'][validIdx],fileName)


def calcAndVis3DDescriptorsWithRot(network,test_data,dataManager=None,fileName=None):

    print("numSamples {}, batch_size {}".format(test_data.numSamples,test_data.batchSize))

    res = network.computeDescriptors(test_data,dataManager,batch_size=test_data.batchSize)
    # throw away results for invalid samples (that were added to fill up minibatches)
    validIdx = test_data.y >= 0
    print("test_data.x.shape {}, num valid {}".format(test_data.numSamples,numpy.sum(validIdx)))
    vis3DDescriptorsWithRot(res[validIdx],test_data.y[validIdx],test_data.sampleInfo['rots'][validIdx],fileName)
    
        
def vis3DDescriptorsWithRot(descrs,labels,rots,fileName=None,tikzFileName=None):

    #cIdx = test_data.y == 0
    #sIdx = test_data.y == 1
    #tIdx = test_data.y == 2  
    
    numObjs = len(numpy.unique(labels))
    #print("numObjs ",numObjs)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if numObjs < 6:
        markers = ['o','s','^','d','v','^','<','>','p','h','8','*','x','+']
    else:
        markers = ['o','.','s','^','d','v','^','<','>','p','h','8','*','x','+']
    #colors = ['b','r','g']
    
    if numObjs > len(markers):
        print "Warning, not enough markers"
    
    numPts = descrs.shape[0]

    x = rots[:,0]
    y = rots[:,1]
    z = -rots[:,2]
    az = numpy.arctan2(y,x).reshape(numPts,1)
    sq = numpy.sqrt(x**2 + y**2)
    el = numpy.arctan2(z,sq).reshape(numPts,1)
    
    h = (az/(2*numpy.pi))+0.5
    s = 1 - el*2/numpy.pi
    v = numpy.ones((numPts,1))
    hsv = numpy.concatenate((h,s,v),axis=1)    
    colors = numpy.array([colorsys.hsv_to_rgb(c[0],c[1],c[2]) for c in hsv])
    
    for i in range(numObjs):
        idx = labels == i
        if descrs.shape[1] > 2:
            z = descrs[idx,2]
        else:
            z = numpy.zeros((numpy.sum(idx),))
        ax.scatter3D(descrs[idx,0],descrs[idx,1],z,c=colors[idx], marker=markers[i])
        #for j in (idx.nonzero()[0]):
        #    ax.scatter3D(descrs[j,0],descrs[j,1],descrs[j,2],c=colors[j], marker=markers[i])
    
    if fileName==None:
        plt.show(block=False)
    else:
        plt.savefig(fileName)      
        
    if tikzFileName is not None:
        tikz_save(tikzFileName, fig, figurewidth='\\figurewidth', figureheight='\\figureheight',show_info=False)        


def showLinemodFrame(frame,title="image_with_center"):
    lcam = LinemodCam()
    mat = frame.pose.getMatrix()
    worldPt0 = numpy.array([0,0,0,1.])
    worldPt1 = numpy.array([0,0,-5,1.])
    camPt0 = numpy.dot(mat,worldPt0)
    camPt1 = numpy.dot(mat,worldPt1)
    #print("camPt",camPt)
    screenPt0 = lcam.worldToScreen(camPt0)
    screenPt1 = lcam.worldToScreen(camPt1)
    print(worldPt0," -> ",screenPt0)
    print(worldPt1," -> ",screenPt1)
    
    #print("rcp {}".format(frame.pose.relCamPos))
   
    im = numpy.copy(frame.img)
    im = im+0.5
    print(im.shape)
    for pt,col in zip([screenPt0,screenPt1],[[255,255,0],[0,255,0]]):
        mnx = numpy.maximum(pt[0]-3,0)
        mxx = numpy.minimum(pt[0]+4,im.shape[1]-1)
        mny = numpy.maximum(pt[1]-3,0)
        mxy = numpy.minimum(pt[1]+4,im.shape[0]-1)
        mx = numpy.minimum(numpy.maximum(pt[0],mnx),mxx)
        my = numpy.minimum(numpy.maximum(pt[1],mny),mxy)
        im[my,mnx:mxx] = numpy.array(col)
        im[mny:mxy,mx] = numpy.array(col)
        
    rectSize2 = 20./2.#28/2.
#     coords = [[[1,0,1],[-1,0,1],[-1,0,-1],[1,0,-1]], [[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]], [[0,1,1],[0,-1,1],[0,-1,-1],[0,1,-1]]]
#     colors = [[255,0,0],[0,255,0],[0,0,255]]
#     for coord,color in zip(coords,colors):
#         rectPt1 = numpy.round(lcam.worldToScreen(numpy.dot(mat,worldPt1+numpy.array([rectSize2*coord[0][0],rectSize2*coord[0][1],rectSize2*coord[0][2],0])))).astype(int)
#         rectPt2 = numpy.round(lcam.worldToScreen(numpy.dot(mat,worldPt1+numpy.array([rectSize2*coord[1][0],rectSize2*coord[1][1],rectSize2*coord[1][2],0])))).astype(int)
#         rectPt3 = numpy.round(lcam.worldToScreen(numpy.dot(mat,worldPt1+numpy.array([rectSize2*coord[2][0],rectSize2*coord[2][1],rectSize2*coord[2][2],0])))).astype(int)
#         rectPt4 = numpy.round(lcam.worldToScreen(numpy.dot(mat,worldPt1+numpy.array([rectSize2*coord[3][0],rectSize2*coord[3][1],rectSize2*coord[3][2],0])))).astype(int)
#         cv2.line(im,tuple(rectPt1),tuple(rectPt2),color=color)
#         cv2.line(im,tuple(rectPt2),tuple(rectPt3),color=color)
#         cv2.line(im,tuple(rectPt3),tuple(rectPt4),color=color)
#         cv2.line(im,tuple(rectPt4),tuple(rectPt1),color=color)
    coords = [[[ 0, 1, 1],[ 0,-1, 1]],  # x-plane
              [[ 0,-1, 1],[ 0,-1,-1]],
              [[ 0,-1,-1],[ 0, 1,-1]],
              [[ 0, 1,-1],[ 0, 1, 1]],
              [[ 1, 0, 1],[-1, 0, 1]],  # y-plane
              [[-1, 0, 1],[-1, 0,-1]],
              [[-1, 0,-1],[ 1, 0,-1]],
              [[ 1, 0,-1],[ 1, 0, 1]], 
              [[ 1, 1, 0],[-1, 1, 0]],  # z-plane
              [[-1, 1, 0],[-1,-1, 0]],
              [[-1,-1, 0],[ 1,-1, 0]],
              [[ 1,-1, 0],[ 1, 1, 0]],
              [[ 0, 0, 0],[ 1, 0, 0]],  # x axis
              [[ 0, 0, 0],[ 0, 1, 0]],   # y axis
              [[ 0, 0, 0],[ 0, 0, 1]]]  # z axis
    
    colors = [[0,0,255],[0,0,255],[0,0,255],[0,0,255],
              [0,255,0],[0,255,0],[0,255,0],[0,255,0],
              [255,0,0],[255,0,0],[255,0,0],[255,0,0],
              [0,0,255],[0,255,0],[255,0,0]]
    
    for coord,color in zip(coords,colors):
        pt1 = numpy.round(lcam.worldToScreen(numpy.dot(mat,worldPt1+numpy.array([rectSize2*coord[0][0],rectSize2*coord[0][1],rectSize2*coord[0][2],0])))).astype(int)
        pt2 = numpy.round(lcam.worldToScreen(numpy.dot(mat,worldPt1+numpy.array([rectSize2*coord[1][0],rectSize2*coord[1][1],rectSize2*coord[1][2],0])))).astype(int)
        cv2.line(im,tuple(pt1),tuple(pt2),color=color)

    
    # paint rectangle faceing the camera
    
    

    cv2.imshow(title,im)
    
    
def visBatchedDatasetCamPoses(data_set):
    r = data_set.sampleInfo['rots']
    uLab = numpy.setdiff1d(numpy.unique(data_set.y),numpy.array(-1),assume_unique=True)
    print("labels: {}".format(uLab))
    numClasses = len(uLab)
    nRows = numpy.round(numpy.sqrt(numClasses/1.4)).astype(numpy.int)
    nCols = int(numpy.ceil(float(numClasses)/float(nRows)))

    fig = plt.figure()
    for i in xrange(numClasses):
        l = uLab[i]
        ax = fig.add_subplot(nRows,nCols,i+1, projection='3d', aspect='equal')
        rl = r[data_set.y == l,:]
        #rl = rl + numpy.tile(numpy.array([l*200.,0.,0.]),(rl.shape[0],1))
        #print("rl.shape {}".format(rl.shape))
        ax.plot(rl[:,0],rl[:,1],rl[:,2],'.')
        ax.set_aspect('equal','box')
        
        # plot cube around 
        mnx = numpy.min(rl[:,0])
        mxx = numpy.max(rl[:,0])
        mny = numpy.min(rl[:,1])
        mxy = numpy.max(rl[:,1])
        mnz = numpy.min(rl[:,2])
        mxz = numpy.max(rl[:,2])
        #print("mn/mx x {},{}; y {},{}; z {},{}".format(mnx,mxx,mny,mxy,mnz,mxz))
        cx = (mxx+mnx)/2.
        cy = (mxy+mny)/2.
        cz = (mxz+mnz)/2.
        print("c {},{},{}".format(cx,cy,cz))
        d = numpy.max([mxx-mnx,mxy-mny,mxz-mnz]) / 2.
        #ax.plot(numpy.array([cx]),numpy.array([cy]),numpy.array([cz]),'go')
        ax.plot(numpy.array([cx-d,cx-d,cx-d,cx-d,cx+d,cx+d,cx+d,cx+d]),
                numpy.array([cy-d,cy-d,cy+d,cy+d,cy-d,cy-d,cy+d,cy+d]),
                numpy.array([cz-d,cz+d,cz-d,cz+d,cz-d,cz+d,cz-d,cz+d]),'r+')
        
    

def visBatchedTrainDatasetCamPoses(train_set):
    print("tidx.shape {}".format(train_set.tripletIdx.shape))
    r = train_set.sampleInfo['rots']
    uLab = numpy.setdiff1d(numpy.unique(train_set.y),numpy.array(-1),assume_unique=True)
    numClasses = len(uLab)
    nRows = numpy.round(numpy.sqrt(numClasses/1.4)).astype(numpy.int)
    nCols = int(numpy.ceil(float(numClasses)/float(nRows)))
    nSamp = len(train_set.y)
    batchSize = train_set.batchSize
    numBatches = nSamp/batchSize
    nTriplets = numBatches* train_set.nTripletsPerBatch
    batchOffsets = (numpy.arange(nTriplets)/train_set.nTripletsPerBatch)*batchSize
    trIdx = train_set.tripletIdx + numpy.tile(batchOffsets.reshape((nTriplets,1)),(1,3)) 
    print("labels: {}".format(uLab))
    fig = plt.figure()
    for i in xrange(numClasses):
        l = uLab[i]
        
        tidx0 = trIdx[:,0]
        print(tidx0.shape)
        print(batchOffsets.shape)
        tidx = trIdx[train_set.y[tidx0] == l,:]
        
        ax = fig.add_subplot(nRows,nCols,i+1, projection='3d', aspect='equal')
        rotSamp = r[numpy.unique(tidx[:,0]),:]
        rotTmpl = r[numpy.unique(tidx[:,1]),:]
        #rl = r[train_set.y == l,:]
        ax.plot(rotSamp[:,0],rotSamp[:,1],rotSamp[:,2],'.')
        ax.plot(rotTmpl[:,0]*1.2,rotTmpl[:,1]*1.2,rotTmpl[:,2]*1.2,'.')
        ax.set_aspect('equal','box')
    plt.tight_layout()
    

def visSimsVsDsts(sim,dst,uLab,rowLabels,colLabels,nSimBins = 90,nDstBins = 100,mxDst=None,clip=None,classLabels=None,fileName=None,tikzFileName=None):
    '''
    plot a histogram of dsts vs sims
    '''
    numClasses = len(uLab)
    print("numClasses {}" .format(numClasses))
    #mnDst = numpy.min(dst.flatten())
    #mxDst = numpy.max(dst.flatten())
    mnDst = 0
    if mxDst is None:
        #mxDst = numpy.max(dst.flatten())
        #print("dst.size {}, 97% {} ".format(dst.size,numpy.floor(dst.size*0.97)))
        
        #mxDst = numpy.sort(dst.flatten())[numpy.floor(dst.size*0.95).astype(numpy.int)]  # take the dst at 97%, the remaining 3% will be clipped to the top
        
        mxDst = numpy.max(dst.flatten())
        totalMinBin = 0
        for lab in uLab:
            lsims = sim[rowLabels==lab][:,colLabels==lab].flatten()
            ldsts = dst[rowLabels==lab][:,colLabels==lab].flatten()
            lsims = numpy.arccos(lsims)*180./numpy.pi 
            h,dstEdges,simEdges = numpy.histogram2d(ldsts,lsims, [nDstBins,nSimBins],range=[[mnDst,mxDst],[0.,180.]])
            for c in xrange(h.shape[1]):
                #if c==0:
                #    print("csh")
                #    print(h[:,c])
                csh = numpy.cumsum(h[:,c])
                idx = (csh > csh[-1]*0.95).nonzero()[0]
                if len(idx) > 0:
                    minBin = idx[0]
                    if minBin > totalMinBin:
                        totalMinBin = minBin 
                        #print(dstEdges)
                        #print(idx)
                        #print("minBin {}".format(minBin))
                        newMxDst = dstEdges[minBin]
        mxDst = newMxDst
        #print("newMxDst: {}".format(newMxDst))
        #print("minBin: {}".format(minBin))
        #print("dstEdges: {}".format(dstEdges))
        
             
    print("dst mn/mx {},{}".format(mnDst,mxDst))
    hists = numpy.zeros((numClasses,nDstBins,nSimBins)) 
    #fig = plt.figure()
    #markers = ['o','s','^','p']
    #colors = ['k','r','g','b']    
    dstEdges = []
    simEdges = []
    for lab in uLab:
        lsims = sim[rowLabels==lab][:,colLabels==lab].flatten()
        ldsts = dst[rowLabels==lab][:,colLabels==lab].flatten()
        lsims = numpy.arccos(lsims)*180./numpy.pi 
        #plt.scatter(lsims,ldsts,c=colors[lab],marker=markers[lab])
    
        #prep 2d hist
        h,dstEdges,simEdges = numpy.histogram2d(ldsts,lsims, [nDstBins,nSimBins],range=[[mnDst,mxDst],[0.,180.]])
        if clip is not None:
            ### do not normalize -> clip
            h = numpy.minimum(clip,h)
        else:
            ### normalize rows
            #   - by sum
            #h = h/numpy.tile(numpy.sum(h,axis=1).reshape(nDstBins,1)+1,(1,nSimBins)).astype(numpy.float)
            #   - by max 
            h = h/numpy.tile(numpy.max(h,axis=1).reshape(nDstBins,1)+1,(1,nSimBins)).astype(numpy.float)
            ### normalize columns
            #   - by max 
            #h = h/numpy.tile(numpy.max(h,axis=0).reshape(1,nSimBins)+1,(nDstBins,1)).astype(numpy.float)
            #   - by sum
            #h = h/numpy.tile(numpy.sum(h,axis=0).reshape(1,nSimBins)+1,(nDstBins,1)).astype(numpy.float)
            #   - by number of templates that are expected at this pose diff angle (more tmpl can have sim 90deg than 180deg)
#         ycenters = (simEdges[:-1]+simEdges[1:])/2.
#         rads = numpy.sin(ycenters*numpy.pi/180.)
#         print rads 
#         h = h/numpy.tile(rads.reshape(1,nSimBins),(nDstBins,1))
        hists[lab] = h
        #print("h min/max {},{}".format(numpy.min(h.flatten()),numpy.max(h.flatten())))
        
    #plt.title("sim vs dst")
        
    nRows = numpy.round(numpy.sqrt(numClasses/1.4)).astype(numpy.int)
    nCols = int(numpy.ceil(float(numClasses)/float(nRows)))

    print("# cols/rows {},{}".format(nCols,nRows))

    fig = plt.figure(figsize=(16,12),dpi=100)
    for lab in uLab:
        ax = fig.add_subplot(nRows,nCols,lab+1)
        
        #ax.pcolor(simEdges,dstEdges,hists[lab])

        ax.pcolormesh(hists[lab])
        ticks = numpy.arange(0, nSimBins+1, nSimBins/5)
        if lab/nCols == nRows-1:
            labels = ticks*180/nSimBins
        else:
            labels = ['']*len(ticks)
        plt.xticks(ticks, labels)

        ticks = numpy.arange(0, nDstBins+1, nDstBins/10)
        if numpy.mod(lab,nCols) == 0:
            labels = numpy.round(ticks*mxDst/float(nDstBins),2)
        else:
            labels = ['']*len(ticks) #numpy.round(ticks*mxDst/float(nDstBins),1)
        plt.yticks(ticks, labels)
        
        if numpy.mod(lab,nCols) == 0:
            plt.ylabel("descr dst")
        if lab/nCols == nRows-1:
            plt.xlabel("angle diff")
        
        #plt.xlabel('Hours')
        if classLabels is None:
            plt.title("class {}".format(lab))
        else:
            plt.title(classLabels[lab])

    if fileName is not None:
        if isinstance(fileName,list):
            for fn in fileName:
                plt.savefig(fn)
        else:  
            plt.savefig(fileName)            
    if tikzFileName is not None:
        #tikz_save(filepath, figure, encoding, figurewidth, figureheight, textsize, tex_relative_path_to_data, strict, draw_rectangles, wrap, extra, show_info)            
        tikz_save(tikzFileName, fig, figurewidth='\\figurewidth', figureheight='\\figureheight',show_info=False)        



def plotAngleErrors(sim,dst,maxAngle=40,maxK=25,kStep=3,fileName=None,tikzFileName=None,pklFileName=None,showPlot=True):
    '''
    plot error in angle of the closest templates
    
    * accuracy when matching to closest tmpl and allowing for angle error x
    * accuracy when matching to the best tmpl among the first k and allowing for angle error x
    '''
    def getGTAngleErrors(sim):
        # angle error of the perfect template
        maxSimIdx = numpy.argmax(sim, axis=1)
        perfAngleErr = numpy.arccos(numpy.maximum(sim[numpy.arange(sim.shape[0]),maxSimIdx],-1.))*180./numpy.pi
        return perfAngleErr    
    
    def getAngleErrors(sim,dst,k=1):
        
        #wrong
        #maxSimIdx = numpy.argmax(sim, axis=1)
        #return dst[:,maxSimIdx]
        
        nSamp = sim.shape[0]
        
        #minDstIdx0 = numpy.argmin(dst, axis=1)
        #angleErr0 = numpy.arccos(numpy.maximum(sim[numpy.arange(nSamp),minDstIdx0],-1.))*180./numpy.pi
    
        minDstIdx = numpy.argsort(dst, axis=1)[:,:k]
        
    #     if k == 1:
    #         if (minDstIdx.flatten() != minDstIdx0).any():
    #             print("minDstIdx.shape {}, minDstIdx0.shape {}".format(minDstIdx.shape,minDstIdx0.shape))
    #             print minDstIdx
    #             print minDstIdx0
            
        angleErr = numpy.zeros((k,nSamp),dtype=numpy.float)
        for i in xrange(k):
            angleErr[i,:] = numpy.arccos(numpy.maximum(sim[numpy.arange(nSamp),minDstIdx[:,i].flatten()],-1.))*180./numpy.pi
        if k > 1:
            angleErr = numpy.min(angleErr,axis=0)    
        
    #     if k == 1:
    #         assert numpy.allclose(angleErr,angleErr0), "{}\n\n{}".format(angleErr,angleErr0) 
        
        return angleErr 

    # accuracy when matching to the best tmpl among the first k and allowing for angle error x
    perfErrs = getGTAngleErrors(sim)    
    gtHst,_ = numpy.histogram(perfErrs, 180,range=(0,180))
    gtCumHst = numpy.cumsum(gtHst)
    gtCumHst = gtCumHst.astype(numpy.float) / float(gtCumHst[-1])
    gtPerfIdx = numpy.min((gtCumHst >= 1.).nonzero())

    ks = numpy.arange(start=1,stop=maxK,step=kStep)
    #maxK = 500
    #ks = numpy.arange(start=1,stop=maxK,step=20)

    h = numpy.arange(len(ks))/float(len(ks))
    print("h {}".format(h))
    colors = getRGBFromHue(h)    

    lineData = []
    lineLabels = []
    lineWidth = []
    lineStyle = []
    for k in ks:
        errs = getAngleErrors(sim,dst,k)
        hst,_ = numpy.histogram(errs, 180,range=(0,180))
        cumHst = numpy.cumsum(hst)
        cumHst = cumHst.astype(numpy.float) / float(cumHst[-1])
        lineData.append((cumHst,k))
        lineLabels.append("k = {}".format(k))
        lineWidth.append(1)
        lineStyle.append('-')
        

    plotAngleErrorsFig(lineData,lineLabels,colors,lineStyle,lineWidth,gtPerfIdx,gtCumHst,maxAngle,fileName=fileName,tikzFileName=tikzFileName,showPlot=showPlot)
    
    if pklFileName is not None:
        with open(pklFileName,'wb') as f:
            cPickle.dump({'lineData':lineData,'lineLabels':lineLabels,'colors':colors,'lineStyle':lineStyle,'lineWidth':lineWidth,'gtPerfIdx':gtPerfIdx,'gtCumHst':gtCumHst,'maxAngle':maxAngle},f,protocol=cPickle.HIGHEST_PROTOCOL)    
    
    
def plotAngleErrorsFig(lineData,lineLabels,colors,lineStyle,lineWidth,gtPerfIdx,gtLine,maxAngle,markers=None,title=None,fileName=None,tikzFileName=None,showPlot=True):

    fig = plt.figure()
    lineHandles = []
    line, = plt.plot(gtLine,'b-.',linewidth=2)
    plt.plot([gtPerfIdx,gtPerfIdx],[0.,1.],'r-.',label="") # used to be "gt", now removed
    #lineHandles.append(line)

    plt.gca().set_color_cycle(colors)

    for i in xrange(len(lineData)):
        ld = lineData[i][0]
        if markers is None:
            line, = plt.plot(ld, linestyle=lineStyle[i], label=lineLabels[i], linewidth=lineWidth[i])
        else:
            line, = plt.plot(ld, linestyle=lineStyle[i], marker=markers[i], label=lineLabels[i], linewidth=lineWidth[i])
            line.set_markevery(5)
        lineHandles.append(line)
        
    plt.xlabel("max. angle diff")
    plt.ylabel("accuracy")
    plt.legend(lineHandles,lineLabels,loc='lower right')
    plt.xlim(0,maxAngle)
    plt.gca().grid(True,which="both")
    if title is not None:
        plt.title(title)
    if showPlot:
        plt.show(block=False) 
    
    if fileName is not None:
        if isinstance(fileName,list):
            for fn in fileName:
                plt.savefig(fn)
        else:  
            plt.savefig(fileName)            
    if tikzFileName is not None:
        #tikz_save(filepath, figure, encoding, figurewidth, figureheight, textsize, tex_relative_path_to_data, strict, draw_rectangles, wrap, extra, show_info)            
        tikz_save(tikzFileName, fig, figurewidth='\\figurewidth', figureheight='\\figureheight',show_info=False)
    
    return fig


def visClosestTmpls(testdata_set, tmpl_set, dst, fileName=None, showPlot=True):
    '''
    Show a random set of test images from testdata_set and its closest templates according to dst
    
    dst: distance of descriptors of test samples and templates (#test x #tmpl)
    '''

    nSamp = 10
    nTmpl = 10
    b = 2
    
    if isinstance(testdata_set.x,list):
        img = numpy.concatenate([x[0] for x in testdata_set.x],axis=0)
    else:
        img = testdata_set.x[0]
    print("img.shape {}".format(img.shape)) 
    nChan,h,w = img.shape
    if nChan == 1:
        dbgImg = numpy.zeros((h*nSamp,w+b+w*nTmpl),dtype=img.dtype)
    elif nChan == 3:
        dbgImg = numpy.zeros((h*nSamp,w+b+w*nTmpl,nChan),dtype=img.dtype)
    else:
        dbgImg = numpy.zeros((h*nSamp,2*w+b+2*w*nTmpl,3),dtype=img.dtype)
    
    print("h,w {}".format(h,w))
    
    #validSampIdx, = (testdata_set.y >= 0).nonzero()
    # choose wisely
    validSampIdx, = numpy.in1d(testdata_set.y,numpy.array([0,2,3,4,5,6,8,9,11,12,14])).nonzero()
    dst = dst[validSampIdx]
    
    numTestSamples = len(validSampIdx)
    randIdx = numpy.random.choice(numTestSamples,nSamp,replace=False)
    if isinstance(testdata_set.x,list):
        #for x in testdata_set.x:
        #    print("shapey shape shape {}".format(x[validSampIdx[randIdx]].shape)) 
        testdata = numpy.concatenate([x[validSampIdx[randIdx]] for x in testdata_set.x],axis=1)
    else:
        testdata = testdata_set.x[validSampIdx[randIdx]] 
    
    if isinstance(tmpl_set.x,list):
        tmplData = numpy.concatenate([x[tmpl_set.y >= 0] for x in tmpl_set.x],axis=1)
    else:
        tmplData = tmpl_set.x[tmpl_set.y >= 0]
    
    for j in range(nSamp):
        sampIdx = randIdx[j]
        
        if nChan == 1:
            img = testdata[j]/2. + 0.5
            dbgImg[h*j:h*(j+1),0:w] = img
        elif nChan == 3:
            img = numpy.rollaxis(numpy.rollaxis( testdata[j] + 0.5 , 2),2) 
            dbgImg[h*j:h*(j+1),0:w,:] = img
        else:
            img = numpy.rollaxis(numpy.rollaxis( testdata[j] , 2),2)
            rgbImg = img[:,:,0:3] + 0.5
            dbgImg[h*j:h*(j+1),0:w,:] = rgbImg
            dImg = img[:,:,3]/2. + 0.5
            dImg = numpy.tile(dImg.reshape((h,w,1)),(1,1,3))
            dbgImg[h*j:h*(j+1),w:2*w,:] = dImg
        
        #nChan,h,w = img.shape
        
        # 10 tmpls with closest descriptor:
        minDstIdx = numpy.argsort(dst[sampIdx,:])
        
        for i in range(nTmpl):
            if nChan == 1:
                l = w+b+w*i
                tmplImg = tmplData[minDstIdx[i]]/2. + 0.5
                dbgImg[h*j:h*(j+1),l:l+w] = tmplImg
            elif nChan == 3:
                l = w+b+w*i
                tmplImg = numpy.rollaxis(numpy.rollaxis( tmplData[minDstIdx[i]] + 0.5, 2), 2)
                dbgImg[h*j:h*(j+1),l:l+w,:] = tmplImg
            else:
                l = 2*w+b+2*w*i
                tmplImg = numpy.rollaxis(numpy.rollaxis( tmplData[minDstIdx[i]], 2), 2)
                rgbImg = tmplImg[:,:,0:3] + 0.5
                dImg = tmplImg[:,:,3]/2. + 0.5
                dImg = numpy.tile(dImg.reshape((h,w,1)),(1,1,3))
                dbgImg[h*j:h*(j+1),l:l+w,:] = rgbImg
                #print("{},{}; {},{}"format(h*j,h*(j+1),l+w,l+w*2))
                dbgImg[h*j:h*(j+1),l+w:l+w*2,:] = dImg
            
    if showPlot:
        cv2.imshow("testAndClosestTmpl",dbgImg)        
    
    if fileName is not None:
        cv2.imwrite(fileName,(dbgImg*255).astype(numpy.uint8))



def visWrongestClosestTmpls(testdata_set, tmpl_set, dst, sim, fileName=None, showPlot=True):
    '''
    Show a set of images from testdata_set for which the closest template according to dst is not very similar according to sim
    
    dst: distance of descriptors of test samples and templates (#test x #tmpl)
    '''

    border = 2
    
    nRows = 10 
    nCols = 10
    nSamp = nRows*nCols
    
    if isinstance(testdata_set.x,list):
        img = numpy.concatenate([x[0] for x in testdata_set.x],axis=0)
    else:
        img = testdata_set.x[0]
    print("img.shape {}".format(img.shape)) 
    nChan,h,w = img.shape

    if nChan == 4:
        w *= 2

    if nChan == 1:
        dbgImg = numpy.zeros(((h+border)*nRows,(2*w+border)*nCols),dtype=img.dtype)
    else:
        dbgImg = numpy.zeros(((h+border)*nRows,(2*w+border)*nCols,3),dtype=img.dtype)

    
    print("h,w {}".format(h,w))
    
    validSampIdx, = (testdata_set.y >= 0).nonzero()
    numTestSamples = len(validSampIdx)
    
    # for every test sample idx of the closest tmpl    
    minDstIdx = numpy.argmin(dst,axis=1)
    # for every test sample similarity of the clostest tmpl
    minDstSim = sim[numpy.arange(numTestSamples),minDstIdx]
    # 
    minSimIdx = numpy.argsort(minDstSim)
    
    minSimIdx = minSimIdx[:nSamp]
    
    print(">>        minDstSim[minSimIdx]")  
    print(minDstSim[minSimIdx])
        
    if isinstance(testdata_set.x,list):  
        ax = 1 if len(validSampIdx[minSimIdx]) > 1 else 0
        testdata = numpy.concatenate([x[validSampIdx[minSimIdx]] for x in testdata_set.x],axis=ax)          
    else:
        testdata = testdata_set.x[validSampIdx[minSimIdx]] 
    #origSamplIdx = testdata_set.sampleInfo['sampIdx'][validSampIdx[minSimIdx]]
    if isinstance(tmpl_set.x,list):
        tmplData = numpy.concatenate([x[tmpl_set.y >= 0] for x in tmpl_set.x],axis=1)
    else:
        tmplData = tmpl_set.x[tmpl_set.y >= 0]

    for j in range(nSamp):
        sampIdx = minSimIdx[j]

        cx = numpy.mod(j,nCols)
        cy = numpy.floor(j/nCols)
        px = cx*(2*w+border)
        py = cy*(h + border)
        
        if nChan == 1:
            img = testdata[j]/2. + 0.5
            dbgImg[py:py+h,px:px+w] = img
        elif nChan == 3:
            img = numpy.rollaxis( numpy.rollaxis(testdata[j] + 0.5, 2),2)
            dbgImg[py:py+h,px:px+w,:] = img
        else:
            img = numpy.rollaxis(numpy.rollaxis( testdata[j] , 2),2)
            rgbImg = img[:,:,0:3] + 0.5
            dbgImg[py:py+h,px:px+w/2,:] = rgbImg
            dImg = img[:,:,3]/2. + 0.5
            dImg = numpy.tile(dImg.reshape((h,w/2,1)),(1,1,3))
            dbgImg[py:py+h,px+w/2:px+w,:] = dImg
            
        #nChan,h,w = img.shape
        
        tmplIdx = minDstIdx[sampIdx] 
        
        if nChan == 1:
            tmplImg = tmplData[tmplIdx]/2. + 0.5
            dbgImg[py:py+h,px+w:px+2*w] = tmplImg
        elif nChan == 3:
            tmplImg = numpy.rollaxis(numpy.rollaxis( tmplData[tmplIdx] + 0.5, 2), 2)
            dbgImg[py:py+h,px+w:px+2*w,:] = tmplImg
        else:
            tmplImg = numpy.rollaxis(numpy.rollaxis( tmplData[tmplIdx], 2), 2)
            rgbImg = tmplImg[:,:,0:3] + 0.5
            dImg = tmplImg[:,:,3]/2. + 0.5
            dImg = numpy.tile(dImg.reshape((h,w/2,1)),(1,1,3))
            dbgImg[py:py+h,px+w:px+3*w/2,:] = rgbImg
            dbgImg[py:py+h,px+3*w/2:px+2*w,:] = dImg
            
        
        #print("orig samp idx {}: {}".format(j,origSamplIdx[j]))
            
    if showPlot:
        cv2.imshow("LeastSimTestAndClosestTmpl",dbgImg)
            
    if fileName is not None:
        #print("depth: {}".format(dbgImg.dtype))
        #print("mn/mx: {},{}".format(numpy.min(dbgImg),numpy.max(dbgImg)))
        cv2.imwrite(fileName,(dbgImg*255).astype(numpy.uint8))


def sampleToPatch(data):
    
    nChan,h,w = data.shape
    
    if nChan == 1:
        # grey
        resPatch = numpy.squeeze(data)/2. + 0.5
    elif nChan == 3:
        # rgb
        resPatch = numpy.rollaxis(numpy.rollaxis( data + 0.5, 2), 2)       
    elif nChan == 4:
        # rgb + grey
        resPatch = numpy.zeros((h,w*2,3))
        tmpImg = numpy.rollaxis(numpy.rollaxis( data, 2), 2)
        rgbImg = tmpImg[:,:,0:3] + 0.5
        dImg = tmpImg[:,:,3]/2. + 0.5
        dImg = numpy.tile(dImg.reshape((h,w,1)),(1,1,3))
        resPatch[:,0:w,:] = rgbImg
        resPatch[:,w:2*w,:] = dImg
        print("mn/mx rgb {},{}".format(numpy.min(rgbImg),numpy.max(rgbImg)))
        print("mn/mx dpt {},{}".format(numpy.min(dImg),numpy.max(dImg)))
                
    return resPatch
