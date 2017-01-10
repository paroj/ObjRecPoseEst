'''
Created on May 12, 2015

@author: wohlhart
'''
import theano.tensor as T
from tnetcore.network import NetworkParams, Network
from tnetcore.util import readCfgParam, readCfgIntNoneListParam,\
    readCfgStrListParam
from tnetcore.layers.convlayer import ConvLayerParams
from tnetcore.layers.poollayer import PoolLayerParams
from tnetcore.layers.actflayer import ActivationFunctionLayerParams
from tnetcore.layers.hiddenlayer import HiddenLayerParams
from tnetcore.layers.flattenlayer import FlattenLayerParams
from tnetcore.layers.catlayer import CatLayerParams
import re
import pydot
import collections
import logging

class DagNetworkParams(NetworkParams):

    def __init__(self,inputDim=None):
        
        self.nwClass = DagNetwork
        
        self._inputDim = inputDim
        if inputDim is not None:
            self.batch_size = inputDim[0]
        else:
            self.batch_size = None
            
        self._layerClasses = dict()
        self.registerLayerClass('conv',ConvLayerParams)
        self.registerLayerClass('pool',PoolLayerParams)
        self.registerLayerClass('actf',ActivationFunctionLayerParams)
        self.registerLayerClass('hidden',HiddenLayerParams)
        self.registerLayerClass('flatten',FlattenLayerParams)
        self.registerLayerClass('cat',CatLayerParams)
    
        self.layerCfgs = []
        self.layerCfgsDict = dict() 
        self.inputLayerCfgs = []
        self._outputLayerCfg = None        
    
    def registerLayerClass(self,name,cls):
        self._layerClasses[name] = cls        
            
    def getLayerParamsClassFromTypeName(self,typeName):
        if self._layerClasses.has_key(typeName):
            return self._layerClasses[typeName]
        else:
            raise ValueError("Unknown layer type: '{}'".format(typeName))    
                    
                    
    def initFromConfig(self,cfg,networkKey='net'):
        
        self._inputDim = readCfgIntNoneListParam(cfg, networkKey, 'inputDim', self._inputDim)
        
        inputDims = self._inputDim
        if (inputDims is not None) and (not isinstance(inputDims[0],collections.Iterable)):
            # single list of dimensions; only one input -> make it a list for easier checking later
            inputDims = [inputDims]
        
        if inputDims is not None:
            self.batch_size = inputDims[0][0] 
        else:
            self.batch_size = None
        #print("DAGNET initFromConfig: batch_size = {}".format(self.batch_size))
        
        layerNames = readCfgStrListParam(cfg, networkKey, 'layers', [])
        if len(layerNames) == 0:
            raise ValueError("Cannot create a DAGNetwork without a 'layers' definition")
        
        self.layerCfgs = []
        self.layerCfgsDict = dict() 

        for layerName in layerNames:
            layerName = layerName.strip()
            if not cfg.has_section(layerName):
                raise ValueError("There is no definition for layer '{}' in the config".format(layerName))
            
            typeName = readCfgParam(cfg, layerName, 'layerType', '')
            LayerParamsClass = self.getLayerParamsClassFromTypeName(typeName)
            
            # assemble input dims
            layerInputNames = readCfgStrListParam(cfg, layerName, 'inputs', [])
            assert layerInputNames, "No inputs given for layer {}".format(layerName) 
            for iln in layerInputNames:
                m = re.match("input\[(\d+)\]",iln)
                if m is not None:
                    # input is an input variable of the network
                    inputVarNum = int(m.group(1))
                    if inputVarNum < len(inputDims):
                        inputDim = inputDims[inputVarNum]
                    else:
                        print("We have no input dim for input var {} (layer {})".format(inputVarNum,layerName))
                else:
                    # input is the output of (a) different layer(s)
                    # check if they are all available
                    assert all([ln in self.layerCfgsDict for ln in layerInputNames]), "ERROR: not all input layers in dictionary {} (layer '{}')".format(layerInputNames,layerName)
                    inputDim = [self.layerCfgsDict[iln].outputDim for iln in layerInputNames]
                    if len(inputDim) == 1:
                        inputDim = inputDim[0]
            
            print('Layer {} - inputDim = {}'.format(layerName,inputDim))
            layerParams = LayerParamsClass(inputDim=inputDim) 
            layerParams.initFromConfig(cfg,layerName)
            self.layerCfgs.append(layerParams)
            
            self.layerCfgsDict[layerName] = layerParams
            
        inputLayersNames = readCfgStrListParam(cfg, networkKey, 'inputLayers',[])
        assert all([iln in self.layerCfgsDict for iln in inputLayersNames]), "Unknown input layer names {}".format(inputLayersNames)
        self.inputLayerCfgs = [self.layerCfgsDict[iln] for iln in inputLayersNames]
        print self.inputLayerCfgs
        assert self.inputLayerCfgs, "ERROR: no input layers defined in DAGNetwork config" 
        
        outputLayerName = readCfgParam(cfg, networkKey, 'outputLayer','')
        assert outputLayerName in self.layerCfgsDict, "ERROR: Unknown output layer: '{}'".format(outputLayerName)
        self.outputLayerCfg = self.layerCfgsDict[outputLayerName]
          
        
            
        
    def addLayerCfg(self,layerCfg):
        self.layerCfgs.append(layerCfg)
        self.layerCfgsDict[layerCfg.name] = layerCfg
        
    @property 
    def outputLayerCfg(self):
        return self._outputLayerCfg

    @outputLayerCfg.setter
    def outputLayerCfg(self,value):
        self._outputLayerCfg = value
        if value is not None:
            self.outputDim = self._outputLayerCfg.outputDim
                
    def __getstate__(self):
        
        #print("DAGNETParams __getstate__")
        state = dict()
        state['nwClass'] = self.nwClass
        state['inputDim'] = self._inputDim
        state['layerCfgs'] = self.layerCfgs
        state['layerCfgsDict'] = self.layerCfgsDict
        state['inputLayerCfgs'] = self.inputLayerCfgs
        state['outputLayerCfg'] = self._outputLayerCfg
        state['batch_size'] = self.batch_size
        state['outputDim'] = self.outputDim

        return state
        
    def __setstate__(self,state):
        
        #print("DAGNETParams __setstate__")
        self.nwClass = state['nwClass']
        self._inputDim = state['inputDim']
        self.layerCfgs = state['layerCfgs']
        self.layerCfgsDict = state['layerCfgsDict']
        self.inputLayerCfgs = state['inputLayerCfgs']
        self._outputLayerCfg = state['outputLayerCfg']
        self.batch_size = state['batch_size']
        self.outputDim = state['outputDim']
        
        
    def debugPlot(self, outfile):
        
        graph = pydot.Dot(graph_type='digraph')
        nodes = dict()
        for layerCfg in self.layerCfgs:
            node = pydot.Node(layerCfg.name)
            graph.add_node(node)
            nodes[layerCfg.name] = node
            
            for inp in layerCfg._inputs:
                if isinstance(inp,int):
                    node = pydot.Node("input[{}]".format(inp))
                    graph.add_node(node)
                    nodes[str(inp)] = node
                    

        for layerCfg in self.layerCfgs:            
            for inp in layerCfg._inputs:
                if isinstance(inp,int):
                    inp = str(inp)
                edge = pydot.Edge(nodes[inp],nodes[layerCfg.name])
                graph.add_edge(edge)
        
        graph.write_png(outfile)
                   
            
class DagNetwork(Network):
    '''
    Directed Acyclic Graph Network
    '''


    def __init__(self, rng, inputVar=None, cfgParams=None):
        '''
        Constructor
        
        @type rng: RandomState
        @type cfgParams: DagNetworkParams 
        @type inputVar: theano.tensor.tensor4
        '''
        if cfgParams is None:
            raise Exception("Cannot create a Network without config parameters (ie. cfgParams==None)")
        
        super(DagNetwork,self).__init__(rng=rng,inputVar=inputVar,cfgParams=cfgParams)
            
            
    def setup(self, rng, inputVar):
        
        logging.getLogger().info("setup network")
        
        if inputVar is None:  # list of input variables
            inputVar = []           
            for inputLayerCfg in self.cfgParams.inputLayerCfgs:
                inputVar.append(T.tensor4('x_'.format(inputLayerCfg.name)))    # input variable
           
        self.inputVar = inputVar
       
        self.cfgParams = self.cfgParams
        
        layersByName = dict()
        layers = []
        params = []
        weights = []
        #for i in xrange(len(cfgParams.layers)):
        for i,layerParams in enumerate(self.cfgParams.layerCfgs):

            layerInput = [] 
            for lIn in layerParams.inputs:
                if isinstance(lIn,int):
                    # its one of the network's inputs
                    if isinstance(inputVar,list):
                        assert lIn >= 0 and lIn < len(inputVar), "Input Var Index {} out of range 0/{}".format(lIn,len(inputVar))
                        layerInput.append(inputVar[lIn])
                    else:
                        assert lIn == 0, "There is only one inputVar to the network, but layer '{}' requests input '{}'".format(layerParams.name,lIn)
                        layerInput.append(inputVar)
                else:
                    # its the output of a different layer
                    assert lIn in layersByName, "input '{}' for layer '{}' is not an existing layer".format(lIn,layerParams.name)
                    olayer = layersByName[lIn]
                    layerInput.append(olayer.output)
                    
            if len(layerInput) == 1:
                layerInput = layerInput[0] 
            #else:
            #    print("layer {} has a list of inputs of len {}".format(layerParams.name,len(layerInput)))

#            if (len(layerParams.inputDim) < 3) and (layerInput.ndim > 2):
#                layerInput = layerInput.flatten(2)

            layer = layerParams.LayerClass(rng,
                                           inputVar=layerInput,
                                           cfgParams=layerParams,
                                           #copyLayer=None if (twin is None) else twin.layers[i],
                                           copyLayer=None,
                                           layerNum = i)
                        
            layers.append(layer)
            params.extend(layer.params)
            weights.extend(layer.weights)

            layersByName[layerParams.name] = layer
            
            
        self.layers = layers
        self.output = layers[-1].output
        self.params = params
        self.weights = weights             
        
        
        
