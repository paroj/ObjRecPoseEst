'''
Created on Feb 18, 2015

@author: wohlhart
'''
import theano
import theano.tensor as T
from tnetcore.layers.base import LayerParams, Layer
from tnetcore.util import readCfgIntNoneListParam, readCfgParam, readCfgFloatParam
from functools import partial

def ReLU(x):
    return x * (x > 0)

def LeakyReLU(x,alpha):
    return T.maximum(x, x*alpha)

class ActivationFunctionLayerParams(LayerParams):
    '''
    Convolution Layer Parameters
    '''
    yaml_tag = u'!ActivationFunctionLayerParams'
    
    def __init__(self, inputDim=None, activation=T.tanh, alpha=0.0):
        '''
        '''
        super(ActivationFunctionLayerParams,self).__init__(inputDim=inputDim,outputDim=None)
        
        self.LayerClass = ActivationFunctionLayer
        
        self._activation = activation
        self._alpha = alpha        
        self.createActfDict()
        self.update()

    def createActfDict(self):
        self._actfs = dict()
        self._actfs["None"] = None
        self._actfs["ReLU"] = ReLU 
        self._actfs["LeakyReLU"] = partial(LeakyReLU,alpha=self._alpha) 
        self._actfs["SoftPlus"] = T.nnet.softplus
        
    def initFromConfig(self,cfg,sectionKey):
        super(ActivationFunctionLayerParams,self).initFromConfig(cfg,sectionKey)
        
        self._inputDim = readCfgIntNoneListParam(cfg,sectionKey,'inputDim',self._inputDim)
        
        self._alpha = readCfgFloatParam(cfg,sectionKey,'alpha',0.05)
        self._actfStr = readCfgParam(cfg,sectionKey,'function','')
        if self._actfs.has_key(self._actfStr):
            self._activation = self._actfs[self._actfStr]
        
        self.update()
        
    @property 
    def activation(self):
        return self._activation          
    
    def update(self):
        '''
        calc outputDim, 
        '''
        self.checkInputDim(expectSingleInput=True)
        
        self._outputDim = self._inputDim
        
    def __getstate__(self):
        
        state = super(ActivationFunctionLayerParams,self).__getstate__()
        state['alpha'] = self._alpha
        state['actfStr'] = self._actfStr
        return state

    def __setstate__(self,state):
        
        super(ActivationFunctionLayerParams,self).__setstate__(state)
        self._alpha = state['alpha'] 
        self._actfStr = state['actfStr']
        self.createActfDict()
        if self._actfs.has_key(self._actfStr):
            self._activation = self._actfs[self._actfStr]
        self.update()
        
    def debugPrint(self,indent=0):
        
        iStr = " "*indent
        print("ActivationFunctionLayer:")
        
        print(iStr + "inputs =          {}".format(self._inputs))
        print(iStr + "inputDim =        {}".format(self._inputDim))
        print(iStr + "outputDim =       {}".format(self._outputDim))
        print(iStr + "activation =      {}".format(self._activation))
                
                
    def initFromYAML(self):
        pass
        
        
class ActivationFunctionLayer(Layer):
    """
    Pool Layer of a convolutional network
    
    copy of LeNetConvPoolLayer from deeplearning.net tutorials
    """

    def __init__(self, rng, inputVar, cfgParams, copyLayer=None, layerNum=None):    
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputVar: theano.tensor.dtensor4
        :param inputVar: symbolic image tensor, of shape image_shape

        :type cfgParams: ConvLayerParams
        """
                
        self.cfgParams = cfgParams
        
        activation  = cfgParams.activation

        self.inputVar = inputVar

        # pass through activation function
        if activation is None:
            self.output = inputVar
        else:
            self.output = activation(inputVar)

        # store parameters of this layer
        self.params = []
        self.weights = []
