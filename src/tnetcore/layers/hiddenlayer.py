'''
Created on 02.05.2014

@author: pwohlhart
'''

import numpy
import theano
import theano.tensor as T
from tnetcore.layers.base import LayerParams, Layer
from tnetcore.util import readCfgIntParam

class HiddenLayerParams(LayerParams):
    
    def __init__(self, inputDim=None,outputDim=None):
        '''
        :type inputDim: int
        :param inputDim: dimensionality of input

        :type outputDim: int
        :param outputDim: number of hidden units
        '''
        super(HiddenLayerParams,self).__init__(inputDim=inputDim,outputDim=outputDim)
        
        self.LayerClass = HiddenLayer
        
        self._inputDim = inputDim
        self._nOutputs = 1
        
        self.update()
        
    @property     
    def wInitOrthogonal(self):
        return self._wInitOrthogonal
    
    @wInitOrthogonal.setter     
    def wInitOrthogonal(self,value):
        self._wInitOrthogonal = value    

    def initFromConfig(self,cfg,sectionKey):
        super(HiddenLayerParams,self).initFromConfig(cfg,sectionKey)
        self._nOutputs = readCfgIntParam(cfg,sectionKey,'outputDim',self._nOutputs)
        self.update()
        
    def update(self):
        '''
        calc outputDim, 
        '''
        if len(self._inputDim) > 2:
            self._inputDim = (self._inputDim[0],numpy.prod(self._inputDim[1:]))
        self._outputDim = (self._inputDim[0],self._nOutputs)
        self.checkOutputDim()
    
    def debugPrint(self,indent=0):
        
        iStr = " "*indent
        print("HiddenLayer:")
        print(iStr + "inputDim =        {}".format(self._inputDim))
        print(iStr + "outputDim =       {}".format(self._outputDim))
        #print(iStr + "wInitOrthogonal = {}".format(self._wInitOrthogonal))
        
    def __getstate__(self):
        state = super(HiddenLayerParams,self).__getstate__()
        state['nOutputs'] = self._nOutputs 
        return state

    def __setstate__(self,state):
        super(HiddenLayerParams,self).__setstate__(state)
        self._nOutputs = state['nOutputs']  
        self.update()           
        
        
class HiddenLayer(Layer):
    def __init__(self, rng, inputVar, cfgParams, copyLayer=None, layerNum=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(inputVar,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputVar: theano.tensor.dmatrix
        :param inputVar: a symbolic tensor of shape (n_examples, n_in)

        :type cfgParams: HiddenLayerParams
        """
        
        self.cfgParams = cfgParams
        
        self.inputVar = inputVar

        n_in = cfgParams.inputDim[1] 
        n_out = cfgParams.outputDim[1]
        activation = cfgParams.activation
#         wInitOrthogonal = cfgParams.wInitOrthogonal 

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        floatX = theano.config.floatX  # @UndefinedVariable
        
        if copyLayer is None:
            
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
                wInitVals = numpy.asarray(rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)), dtype=floatX)
                if activation == T.nnet.sigmoid:
                    wInitVals *= 4
                bInitVals = numpy.zeros((n_out,), dtype=floatX)
                
#             if wInitOrthogonal:
#                 svd = numpy.linalg.svd(W_values)
#                 U = svd[0]
#                 wInitVals = U.T[0:filter_shape[0]].T
#                 wInitVals = numpy.reshape(wInitVals.swapaxes(0,1), filter_shape)

            self.W = theano.shared(value=wInitVals, name='W{}'.format(layerNum), borrow=True)
            self.b = theano.shared(value=bInitVals, name='b{}'.format(layerNum), borrow=True)
            
        else:
            self.W = copyLayer.W
            self.b = copyLayer.b

        #flatInput = inputVar.flatten(2)
        lin_output = T.dot(inputVar, self.W) + self.b
        self.output = lin_output
        
        # parameters of the model
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