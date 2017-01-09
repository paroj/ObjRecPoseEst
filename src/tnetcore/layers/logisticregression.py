'''
Created on 02.05.2014

@author: pwohlhart
'''

import numpy
import theano
import theano.tensor as T
from tnetcore.layers.base import LayerParams, Layer


class LogisticRegressionParams(LayerParams):
    
    def __init__(self,inputDim=None,outputDim=None):
        '''
        :type inputDim: tuple(int,int)
        :param inputDim: number of inputVar units, the dimension of the space in
                     which the datapoints lie

        :type outputDim: tuple(int,int)
        :param outputDim: number of output units, the dimension of the space in which the labels lie
        '''        
        super(LogisticRegressionParams,self).__init__(inputDim=inputDim,outputDim=outputDim)

    def initFromConfig(self,cfg,sectionKey):
        super(LogisticRegressionParams,self).initFromConfig(cfg,sectionKey)
        

class LogisticRegression(Layer):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, rng, inputVar, cfgParams, copyLayer=None):
        """ Initialize the parameters of the logistic regression

        :type inputVar: theano.tensor.TensorType
        :param inputVar: symbolic variable that describes the inputVar of the
                      architecture (one minibatch)


        :type cfgParams: LogisticRegressionParams
        """
        floatX = theano.config.floatX  # @UndefinedVariable
        
        n_in = cfgParams.inputDim[1]
        n_out = cfgParams.outputDim[1]

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
                # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
                #Wvals = numpy.zeros((n_in, n_out),dtype=floatX)
                W_bound = 1 #. / (n_in + n_out)
                wInitVals = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(n_in,n_out)),dtype=floatX)
                bInitVals = numpy.zeros((n_out,),dtype=floatX)
            
            self.W = theano.shared(value=wInitVals,
                                   name='W', 
                                   borrow=True)
            # initialize the biases b as a vector of n_out 0s
            self.b = theano.shared(value=bInitVals,
                                   name='b', 
                                   borrow=True)
        else:
            self.W = copyLayer.W
            self.b = copyLayer.b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(inputVar, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]
        self.weights = [self.W]

        # output descriptor
        self.output = self.p_y_given_x   

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.sum(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        
        
    def saveValues(self,fileName):
        '''
        We just have to store W and b and put that in the cfgParams
          that way we can be reconstructed by creating a new instance and just running through init
        '''
        fn = fileName + ".npz"
        with open(fn,"wb") as f:
            numpy.savez(f,W=self.W.get_value(),b=self.b.get_value())
        self.cfgParams.initValues = fn        
