'''
Created on 02.05.2014

@author: pwohlhart
'''

import numpy
import scipy.stats

import theano
import theano.tensor as T
from theano.tensor import nnet as conv
from tnetcore.layers.base import LayerParams, Layer

from tnetcore.util import readCfgIntNoneListParam, readCfgIntParam, readCfgBooleanParam
import logging

class ConvLayerParams(LayerParams):
    '''
    Convolution Layer Parameters
    '''
    yaml_tag = u'!ConvLayerParams'
    
    def __init__(self, inputDim=None,nFilters=None,filterDim=None,activation=T.tanh,poolsize=None,poolType=0,filter_shape=None,image_shape=None,outputDim=None,wInitMode=0,wInitOrthogonal=False):
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
        super(ConvLayerParams,self).__init__(inputDim,outputDim)
       
        self.LayerClass = ConvLayer
        
        self._inputDim = inputDim
        self._nFilters = nFilters
        self._filterDim = filterDim
        self._filter_shape = filter_shape
        self._image_shape = image_shape
        self._outputDim = outputDim
        self._wInitMode = wInitMode
        self._wInitOrthogonal = wInitOrthogonal
        self.update()


    def initFromConfig(self,cfg,sectionKey):
        super(ConvLayerParams,self).initFromConfig(cfg,sectionKey)
        
        self._inputDim = readCfgIntNoneListParam(cfg,sectionKey,'inputDim',self._inputDim)
        self._nFilters = readCfgIntParam(cfg,sectionKey,'nFilters',self._nFilters)
        self._filterDim = readCfgIntNoneListParam(cfg,sectionKey,'filterDim',self._filterDim)
        #self._filter_shape = filter_shape
        #self._image_shape = image_shape
        #self._outputDim = readCfgIntNoneListParam(cfg,sectionKey,'outputDim',self._outputDim)
        self._wInitMode = readCfgIntParam(cfg,sectionKey,'wInitMode',self._wInitMode)
        self._wInitOrthogonal = readCfgBooleanParam(cfg,sectionKey,'wInitOrthogonal',self._wInitOrthogonal)
        self.update()

    @property 
    def filter_shape(self):
        return self._filter_shape 

    @property 
    def image_shape(self):
        return self._image_shape 

    @property 
    def nFilters(self):
        return self._nFilters 
    
    @nFilters.setter 
    def nFilters(self,value):
        self._nFilters = value
        self.update()
     
    @property 
    def filterDim(self):
        return self._filterDim 
    
    @filterDim.setter 
    def filterDim(self,value):
        self._filterDim = value
        self.update()
        
    @property 
    def poolsize(self):
        return self._poolsize
    
    @property                  
    def wInitMode(self):
        return self._wInitMode

    @wInitMode.setter                  
    def wInitMode(self,value):
        self._wInitMode = value
    
    @property                  
    def wInitOrthogonal(self):
        return self._wInitOrthogonal

    @wInitOrthogonal.setter                  
    def wInitOrthogonal(self,value):
        self._wInitOrthogonal = value    
    

    def update(self):
        '''
        calc image_shape, 
        '''
        if (self._filterDim is None) or (self._inputDim is None):
            return
        
        #assert (len(self._inputDim) > 0) and (not isinstance(self._inputDim[0],(list,tuple))), "Conv Layer Can only deal with single input, but inputDim is %s" % self._inputDim
        self.checkInputDim(expectSingleInput=True)    
        
        self._filter_shape = (self._nFilters,
                              self._inputDim[1],
                              self._filterDim[0],
                              self._filterDim[1])
        self._image_shape = self._inputDim
        
        # TODO: depends on type of convolution (valid, full, same) 
        self._outputDim = (self._inputDim[0],   # batch_size
                           self._nFilters,      # number of kernels
                           (self._inputDim[2] - self._filterDim[0] + 1),   #  output H
                           (self._inputDim[3] - self._filterDim[1] + 1))   #  output W
        
        self.checkOutputDim()
        
    def debugPrint(self,indent=0):
        
        iStr = " "*indent
        print("ConvLayer:")
        
        print(iStr + "inputs =          {}".format(self._inputs))
        print(iStr + "inputDim =        {}".format(self._inputDim))
        print(iStr + "nFilters =        {}".format(self._nFilters))
        print(iStr + "filterDim =       {}".format(self._filterDim))
        print(iStr + "filter_shape =    {}".format(self._filter_shape))
        print(iStr + "image_shape =     {}".format(self._image_shape))
        print(iStr + "outputDim =       {}".format(self._outputDim))
        print(iStr + "wInitMode =       {}".format(self._wInitMode))
        print(iStr + "wInitOrthogonal = {}".format(self._wInitOrthogonal))
                
    def __getstate__(self):
        state = super(ConvLayerParams,self).__getstate__()
        state['nFilters'] = self._nFilters 
        state['filterDim'] = self._filterDim 
        state['filter_shape'] = self._filter_shape
        state['image_shape'] = self._image_shape 
        state['wInitMode'] = self._wInitMode 
        state['wInitOrthogonal'] = self._wInitOrthogonal        
        return state

    def __setstate__(self,state):
        super(ConvLayerParams,self).__setstate__(state)
        self._nFilters = state['nFilters']  
        self._filterDim = state['filterDim']  
        self._filter_shape = state['filter_shape'] 
        self._image_shape = state['image_shape'] 
        self._wInitMode = state['wInitMode']  
        self._wInitOrthogonal = state['wInitOrthogonal']        
        self.update()                
        
class ConvLayer(Layer):
    """
    Pool Layer of a convolutional network
    
    copy of LeNetConvPoolLayer from deeplearning.net tutorials
    """

    #def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), copyLayer=None):
    def __init__(self, rng, inputVar, cfgParams, copyLayer=None, layerNum=None):    
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputVar: theano.tensor.dtensor4
        :param inputVar: symbolic image tensor, of shape image_shape

        :type cfgParams: ConvLayerParams
        """
                
        floatX = theano.config.floatX  # @UndefinedVariable
        
        self.cfgParams = cfgParams
        
        filter_shape = cfgParams.filter_shape
        image_shape = cfgParams.image_shape
        wInitOrthogonal = cfgParams.wInitOrthogonal

        assert image_shape[1] == filter_shape[1]
        self.inputVar = inputVar

        # there are "num inputVar feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))

        bInitVals = numpy.zeros((filter_shape[0],), dtype=floatX)
        
        # initialize weights with random weights
        if not (copyLayer is None):
            self.W = copyLayer.W
        else:
            if cfgParams.initValues is not None:
                if isinstance(cfgParams.initValues,basestring):
                    # load values from npz file
                        with open(cfgParams.initValues,"rb") as f:
                            vals = numpy.load(f)
                            wInitVals = vals['W'] 
                            bInitVals = vals['b']
                else:
                    if isinstance(cfgParams.initValues,dict):
                        wInitVals = cfgParams.initValues['W'] 
                        bInitVals = cfgParams.initValues['b']
                assert isinstance(wInitVals,numpy.ndarray), "Error loading W init values in HiddenLayer {}".format(self.cfgParams.name)
                assert isinstance(bInitVals,numpy.ndarray), "Error loading b init values in HiddenLayer {}".format(self.cfgParams.name)
            else:
                if cfgParams.wInitMode == 0:
                    W_bound = 1. / (fan_in + fan_out)
                    wInitVals = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),dtype=floatX)
                elif cfgParams.wInitMode == 1:
                    W_bound = numpy.sqrt(1. / (fan_in + fan_out))
                    wInitVals = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),dtype=floatX)
                elif cfgParams.wInitMode == 2:  # init
                    
                    wInitVals = numpy.zeros(filter_shape,dtype=floatX)
                    W_bound = numpy.sqrt(1. / (fan_in + fan_out))
                    #nFilters = cfgParams.nFilters
                    #nInputs = cfgParams.inputDim[1]
                    nFilters = filter_shape[0]
                    inputDim = filter_shape[1]
                    filterH = filter_shape[2]
                    filterW = filter_shape[3]
                    
                    filterSize = numpy.minimum(filterH,filterW)
                    mx,my = numpy.meshgrid(numpy.arange(filterH),numpy.arange(filterW),dtype=floatX)
                    
                    d = scipy.stats.norm(0,1)
                    y = d.pdf(numpy.linspace(-2,2,filterW))
                    y = y.reshape((filterW,1))
                    g = y.T*y
                    g = g/numpy.max(g)
                    
                    for i in range(nFilters):
                        for j in range(inputDim):
                            fx = numpy.random.rand()*2-1
                            fy = numpy.random.rand()*2-1
                            f = numpy.sin((mx*fx+my*fy)*2.*numpy.pi/filterSize)
                            fx = numpy.random.rand()*4-2
                            fy = numpy.random.rand()*4-2
                            f += numpy.random.rand() * numpy.sin((mx*fx-my*fy)*2.*numpy.pi/filterSize)
                            f = (f - numpy.min(f))*2*W_bound/(numpy.max(f)-numpy.min(f)) - W_bound                     
                            
                            f = f*g  # gaussian window
                            
                            wInitVals[i,j] = f
                
                
                # try pca of 
                # to create an orthogonal set of filters to start with
                #wInitOrthogonal = True # True ?
                if wInitOrthogonal:
                    wInitVals = numpy.reshape(wInitVals,(filter_shape[0],numpy.prod(filter_shape[1:])))
                    svd = numpy.linalg.svd(wInitVals.T)
                    U = svd[0]
                    wInitVals = U.T[0:filter_shape[0]].T
                    wInitVals = numpy.reshape(wInitVals.swapaxes(0,1), filter_shape)
            
            #wInitVals = numpy.asarray(numpy.ones(filter_shape,dtype=floatX)/(10*(fan_in+fan_out))) 
            self.W = theano.shared(wInitVals, borrow=True, name='convW{}'.format(layerNum))

        # the bias is a 1D tensor -- one bias per output feature map
        if not (copyLayer is None):
            self.b = copyLayer.b
        else:
            self.b = theano.shared(value=bInitVals, borrow=True, name='convB{}'.format(layerNum))

        # convolve inputVar feature maps with filters
        conv_out = conv.conv2d(input = inputVar, 
                               filters = self.W,
                               filter_shape = filter_shape, 
                               image_shape = image_shape)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        
        self.output = out

        # store parameters of this layer
        self.params = [self.W, self.b]
        self.weights = [self.W]


    def saveValues(self,fileName):
        '''
        We just have to store W and b and put that in the cfgParams
          that way we can be reconstructed by creating a new instance and just running through init
        '''
        fn = fileName + ".npz"
        with open(fn,"wb") as f:
            numpy.savez(f,W=self.W.get_value(),b=self.b.get_value())
        self.cfgParams.initValues = fn
        
