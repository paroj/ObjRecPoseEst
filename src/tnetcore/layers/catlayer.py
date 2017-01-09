'''
Created on May 12, 2015

@author: wohlhart
'''
from tnetcore.layers.base import LayerParams, Layer
from tnetcore.util import readCfgIntNoneListParam, readCfgIntParam  # @UnresolvedImport
import theano.tensor as T
import numpy

class CatLayerParams(LayerParams):
    '''
    Concatenation Layer Parameters
    '''
    yaml_tag = u'!CatLayerParams'
    
    def __init__(self, inputDim=None,axis=1):
        '''
        '''
        super(CatLayerParams,self).__init__(inputDim=inputDim,outputDim=None)
        
        self.LayerClass = CatLayer
        
        self._inputDim = inputDim
        self._axis = axis
        self.update()


    def initFromConfig(self,cfg,sectionKey):
        super(CatLayerParams,self).initFromConfig(cfg,sectionKey)
        self._inputDim = readCfgIntNoneListParam(cfg,sectionKey,'inputDim',self._inputDim)
        self._axis = readCfgIntParam(cfg,sectionKey,'axis',self._axis)
        self.update()

    @property 
    def axis(self):
        return self._axis
    
    @axis.setter 
    def axis(self,value):
        self._axis = value
        self.update()
        
    def update(self):
        '''
        calc outputDim
        '''
        if (self._axis is None) or (self._inputDim is None):
            return
        
        #assert len(self._inputDim) > 1 and len(self._inputDim[0]) > 1, "CatLayer needs more than one input"
        self.checkInputDim(expectMultipleInputs=True)
        
        # inputDim is a list of inputDims. check if they agree along the non-cat-axes
        inDim = numpy.array(self._inputDim)
        #print("self._inputDim {}".format(self._inputDim))  
        #print("inDim {}".format(inDim))
        assert inDim.shape[1] == 4, "Wrong inputDim shape {}; each row must have 4 entries (bs,nchan,h,w)".format(inDim.shape) 
        numInputs = inDim.shape[0]
        nonCatAxes = numpy.setdiff1d(numpy.arange(inDim.shape[1]),[self._axis])
        numEqual = numpy.sum(inDim == inDim[0],axis=0)
        assert all(numEqual[nonCatAxes] == numInputs), "the axes along which not to concatenate must be equal, but are (axis={})\n{}".format(self._axis,inDim)

        # outDim is inDim for all non-cat-axes and sum over inDims for the cat axis        
        outDim = inDim[0]
        outDim[self._axis] = sum(inDim[:,self._axis])
        self._outputDim = list(outDim)
        
        self.checkOutputDim()
    
        
    def debugPrint(self,indent=0):
        
        iStr = " "*indent
        print("CatLayer:")
        
        print(iStr + "inputs =          {}".format(self._inputs))
        print(iStr + "inputDim =        {}".format(self._inputDim))
        print(iStr + "axis =        {}".format(self._axis))
        print(iStr + "outputDim =       {}".format(self._outputDim))
                
                
    def __getstate__(self):
        state = super(CatLayerParams,self).__getstate__()
        state['axis'] = self._axis
        return state

    def __setstate__(self,state):
        super(CatLayerParams,self).__setstate__(state)
        self._axis = state['axis'] 
        self.update()
        
        
class CatLayer(Layer):
    """
    Concatenation Layer 
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
        
        axis  = cfgParams.axis

        self.inputVar = inputVar
        
        self.output = T.concatenate(inputVar, axis)

        # store parameters of this layer; has none
        self.params = []
        self.weights = []


