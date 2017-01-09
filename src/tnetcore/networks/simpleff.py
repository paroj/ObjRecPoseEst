'''
Created on Feb 18, 2015

@author: wohlhart
'''
from tnetcore.network import Network, NetworkParams
from tnetcore.layers.convlayer import ConvLayer, ConvLayerParams
from tnetcore.layers.poollayer import PoolLayer, PoolLayerParams  # @UnresolvedImport
from tnetcore.layers.actflayer import ActivationFunctionLayer,\
    ActivationFunctionLayerParams
from tnetcore.layers.hiddenlayer import HiddenLayerParams

from tnetcore.util import readCfgParam, readCfgIntNoneListParam  # @UnresolvedImport
from tnetcore.layers.flattenlayer import FlattenLayerParams  # @UnresolvedImport
from tnetcore.layers.catlayer import CatLayerParams

class SimpleFFNetworkParams(NetworkParams):
    
    def __init__(self, inputDim = None):
        '''
        '''
        self.nwClass = SimpleFFNetwork
                
        self._inputDim = inputDim
        if inputDim is not None:
            self.batch_size = inputDim[0]
        else:
            self.batch_size = None
        
    def initFromConfig(self,cfg,networkKey='net'):
        #typeID,nChan=3,wIn=64,hIn=64,batchSize=128,nOut=3
        #self.readCfgIntParam(cfg, 'net', 'type', default=0)
        layerDef = readCfgParam(cfg, networkKey, 'layers', '')
        
        self._inputDim = readCfgIntNoneListParam(cfg, networkKey, 'inputDim', self._inputDim)
        if self._inputDim is not None:
            self.batch_size = self._inputDim[0]
        else:
            self.batch_size = None
        
        lastLayerOutputDim = self._inputDim 
        
        layerDef = layerDef.strip()
        if layerDef == '':
            raise ValueError("Cannot create a SimpleFFNetwork without a 'layers' definition")
        
        self.layerCfgs = [] 

        layerNames = layerDef.split(',')
        
        for layerName in layerNames:
            layerName = layerName.strip()
            if not cfg.has_section(layerName):
                raise ValueError("There is no definition for layer '{}' in the config".format(layerName))
            
            typeName = readCfgParam(cfg, layerName, 'layerType', '')
            LayerParamsClass = self.getLayerParamsClassFromTypeName(typeName)
            layerParams = LayerParamsClass(inputDim = lastLayerOutputDim) 
            layerParams.initFromConfig(cfg,layerName)
            self.layerCfgs.append(layerParams)
            
            lastLayerOutputDim = layerParams.outputDim
            
        self.outputDim = lastLayerOutputDim
            
        
    def getLayerParamsClassFromTypeName(self,typeName):
        if typeName == 'conv':
            return ConvLayerParams
        if typeName == 'pool':
            return PoolLayerParams
        if typeName == 'actf':
            return ActivationFunctionLayerParams
        if typeName == 'hidden':
            return HiddenLayerParams
        if typeName == 'flatten':
            return FlattenLayerParams
        if typeName == 'cat':
            return CatLayerParams
                
        raise ValueError("Unknown layer type: '{}'".format(typeName))
            
        
    def __getstate__(self):
        
        state = dict()
        state['nwClass'] = self.nwClass
        state['batch_size'] = self.batch_size
        state['layerCfgs'] = self.layerCfgs
        state['outputDim'] = self.outputDim
        return state
        
    def __setstate__(self,state):
        
        self.nwClass = state['nwClass'] 
        self.batch_size = state['batch_size'] 
        self.layerCfgs = state['layerCfgs'] 
        self.outputDim = state['outputDim']         
        

class SimpleFFNetwork(Network):
    '''
    classdocs
    '''


    def __init__(self, rng, inputVar=None, cfgParams=None):
        '''
        Constructor
        '''
        super(SimpleFFNetwork,self).__init__(rng=rng,inputVar=inputVar,cfgParams=cfgParams)
        
        
        