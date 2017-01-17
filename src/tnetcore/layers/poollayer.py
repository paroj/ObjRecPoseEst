'''
Created on Feb 18, 2015

@author: wohlhart
'''
from theano.tensor.signal import pool
from tnetcore.layers.base import LayerParams, Layer
from tnetcore.util import readCfgIntNoneListParam, readCfgIntParam  # @UnresolvedImport

class PoolLayerParams(LayerParams):
    '''
    Convolution Layer Parameters
    '''
    yaml_tag = u'!PoolLayerParams'
    
    def __init__(self, inputDim=None,poolsize=None,poolType=0):
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
        super(PoolLayerParams,self).__init__(inputDim=inputDim,outputDim=None)
                
        self.LayerClass = PoolLayer
        
        self._poolsize = poolsize
        self._poolType = poolType
        self.update()


    def initFromConfig(self,cfg,sectionKey):
        super(PoolLayerParams,self).initFromConfig(cfg,sectionKey)
        self._inputDim = readCfgIntNoneListParam(cfg,sectionKey,'inputDim',self._inputDim)
        self._poolType = readCfgIntParam(cfg,sectionKey,'poolType',self._poolType)
        self._poolsize = readCfgIntNoneListParam(cfg,sectionKey,'poolsize',self._poolsize)
        self.update()

    @property 
    def poolsize(self):
        return self._poolsize
    
    @poolsize.setter 
    def poolsize(self,value):
        self._poolsize = value
        self.update()
        
    @property 
    def poolType(self):
        return self._poolType        


    def update(self):
        '''
        calc outputDim
        '''
        if (self._poolsize is None) or (self._inputDim is None):
            return
        
        # TOOD: depends on pool stride (?)
        self._outputDim = (self._inputDim[0],   # batch_size
                           self._inputDim[1],   # feature maps input
                           self._inputDim[2]/self._poolsize[0],   #  output H
                           self._inputDim[3]/self._poolsize[1])   #  output W
        self.checkOutputDim()
        
    def debugPrint(self,indent=0):
        
        iStr = " "*indent
        print("PoolLayer:")
        
        print(iStr + "inputDim =        {}".format(self._inputDim))
        print(iStr + "poolsize =        {}".format(self._poolsize))
        print(iStr + "poolType =        {}".format(self._poolType))
        print(iStr + "outputDim =       {}".format(self._outputDim))
                
                
    def __getstate__(self):
        state = super(PoolLayerParams,self).__getstate__()
        state['poolsize'] = self._poolsize 
        state['poolType'] = self._poolType 
        return state

    def __setstate__(self,state):
        super(PoolLayerParams,self).__setstate__(state)
        self._poolsize = state['poolsize']  
        self._poolType = state['poolType']  
        self.update()           
        
        
class PoolLayer(Layer):
    """
    Pool Layer of a convolutional network
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
        
        poolsize  = cfgParams.poolsize
        poolType  = cfgParams.poolType

        self.inputVar = inputVar

        if poolType == 0:
            pooled_out = pool.pool_2d(input = inputVar,
                                      ds = poolsize, ignore_border=True)

        self.output = pooled_out

        # store parameters of this layer; has none
        self.params = []
        self.weights = []


