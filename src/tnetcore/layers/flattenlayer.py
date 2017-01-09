'''
Created on Mar 31, 2015

@author: wohlhart
'''
import theano.tensor as T
import numpy
from tnetcore.layers.base import LayerParams, Layer
from tnetcore.util import readCfgIntNoneListParam, readCfgParam


class FlattenLayerParams(LayerParams):
    '''
    Flatten Layer Parameters
    '''
    def __init__(self, inputDim=None):
        '''
        '''
        super(FlattenLayerParams,self).__init__(inputDim=inputDim,outputDim=None)
        
        self.LayerClass = FlattenLayer
        
        self._inputDim = inputDim
        self.update()

    def initFromConfig(self,cfg,sectionKey):
        super(FlattenLayerParams,self).initFromConfig(cfg,sectionKey)
        self.update()
        
    def update(self):
        '''
        calc outputDim, 
        '''
        #self._outputDim = numpy.array([self._inputDim[0],numpy.prod(self._inputDim[1:])])
        self._outputDim = (self._inputDim[0],numpy.prod(self._inputDim[1:]))
        self.checkOutputDim()
        
    def debugPrint(self,indent=0):
        
        iStr = " "*indent
        print("FlattenLayer:")
        
        print(iStr + "inputs =          {}".format(self._inputs))
        print(iStr + "inputDim =        {}".format(self._inputDim))
        print(iStr + "outputDim =       {}".format(self._outputDim))
                
                
        
        
class FlattenLayer(Layer):
    """
    Pool Layer of a convolutional network
    
    copy of LeNetConvPoolLayer from deeplearning.net tutorials
    """

    def __init__(self, rng, inputVar, cfgParams, copyLayer=None, layerNum=None):    
        """
        Flatten output of previous layer such that there is only two dimensions: 
            one data vector per input sample in the minibatch

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputVar: theano.tensor.dtensor4
        :param inputVar: symbolic image tensor, of shape image_shape

        :type cfgParams: ConvLayerParams
        """
                
        self.cfgParams = cfgParams
        
        self.inputVar = inputVar

        self.output = inputVar.reshape(cfgParams.outputDim)

        # store parameters of this layer
        self.params = []
        self.weights = []


