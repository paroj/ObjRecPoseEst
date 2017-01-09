'''
Created on Mar 12, 2015

@author: wohlhart
'''
from theano.tensor.signal import downsample
from tnetcore.layers.base import LayerParams, Layer
from tnetcore.util import readCfgIntNoneListParam, readCfgIntParam, readCfgFloatParam

import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import tnetcore.util
import cv
import cv2

class LCNLayerParams(LayerParams):
    '''
    Local Contrast Normalization Layer Parameters
    '''
    yaml_tag = u'!LCNLayerParams'
    
    def __init__(self, inputDim=None,sigma=3.0):
        '''
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num inputVar feature maps,
                          filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num inputVar feature maps,
                         image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        '''
        super(LCNLayerParams,self).__init__(inputDim=inputDim,outputDim=None)
        
        self.LayerClass = LCNLayer
        
        self._inputDim = inputDim
        self._sigma = sigma
        self.update()


    def initFromConfig(self,cfg,sectionKey):
        super(LCNLayerParams,self).initFromConfig(cfg,sectionKey)
        
        self._inputDim = readCfgIntNoneListParam(cfg,sectionKey,'inputDim',self._inputDim)
        #self._sigma1 = readCfgFloatParam(cfg,sectionKey,'sigma1',self._sigma1)
        #self._sigma2 = readCfgFloatParam(cfg,sectionKey,'sigma2',self._sigma2)
        self._radius = readCfgFloatParam(cfg,sectionKey,'radius',self._radius)
        self.update()

    @property 
    def sigma(self):
        return self._sigma
    
    @sigma.setter 
    def sigma(self,value):
        self._sigma = value
        self.update()

    def update(self):
        '''
        calc outputDim
        '''
        self._outputDim = self._inputDim  # is that right? seems unfinished
    
        
    def debugPrint(self,indent=0):
        
        iStr = " "*indent
        print("LCNLayer:")
        
        print(iStr + "inputDim =        {}".format(self._inputDim))
        print(iStr + "radius =        {}".format(self._sigma))
        print(iStr + "outputDim =       {}".format(self._outputDim))
                
                
    def __getstate__(self):
        state = super(LCNLayerParams,self).__getstate__()
        state['sigma'] = self._sigma 
        return state

    def __setstate__(self,state):
        super(LCNLayerParams,self).__setstate__(state)
        self._sigma = state['sigma']  
        self.update()             
        
class LCNLayer(Layer):
    """
    Local Contrast Normalization Layer
    """

    def __init__(self, rng, inputVar, cfgParams, copyLayer=None, layerNum=None):    
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputVar: theano.tensor.dtensor4
        :param inputVar: symbolic image tensor, of shape image_shape

        :type cfgParams: ConvLayerParams
        """
        self.cfgParams = cfgParams
        
        self.inputVar = inputVar

        sigma = cfgParams.sigma
        inputDim = cfgParams.inputDim
        
        nSig = 2 # sigmas in both directions
        radius = numpy.ceil(nSig*sigma).astype(int)
        patchSize = radius*2 + 1
        
        gauss1d = numpy.exp(-numpy.power( numpy.linspace(-nSig*sigma,nSig*sigma,num=patchSize), 2.) / (2 * numpy.power(sigma, 2.)))
        gaussPatch = numpy.outer(gauss1d,gauss1d)
        gaussPatch = gaussPatch / numpy.sum(gaussPatch)
        #filterValsGauss = numpy.zeros((1,1,patchSize,patchSize))
        #for i in xrange(inputDim[1]): 
        #    filterValsGauss[i,i,:,:] = gaussPatch 
        
        #cv2.imshow("gaussPatch",gaussPatch/numpy.max(gaussPatch))
        #cv2.waitKey() 
        
        filterValsGauss = gaussPatch.reshape((1,1,patchSize,patchSize))
        filters = theano.shared(filterValsGauss.astype(numpy.float32))
        
        inputSq = inputVar*inputVar
        mean_full_channels = []
        sqMean_full_channels = []
        for i in xrange(inputDim[1]):
            mean_full_channels.append(conv.conv2d(inputVar[:,i,:,:].reshape((inputDim[0],1,inputDim[2],inputDim[3])),filters,border_mode='full'))
            sqMean_full_channels.append(conv.conv2d(inputSq[:,i,:,:].reshape((inputDim[0],1,inputDim[2],inputDim[3])),filters,border_mode='full'))
        mean_full = T.concatenate(mean_full_channels, axis=1)
        sqMean_full = T.concatenate(sqMean_full_channels, axis=1)

        mean = mean_full[:,:,radius:-radius,radius:-radius] 
        sqMean = sqMean_full[:,:,radius:-radius,radius:-radius] 

        meanSq = mean*mean
        var = sqMean - meanSq

        self.output = (inputVar - mean) / (T.maximum(var,0.1))
        #self.output = T.maximum(var,0.01)
 
        #self.output = sqMean

        # store parameters of this layer; has none
        self.params = []
        self.weights = []

