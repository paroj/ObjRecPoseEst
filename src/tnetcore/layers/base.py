'''
Created on 08.05.2014

@author: pwohlhart
'''
from tnetcore.util import readCfgStrListParam
import re
import collections

class LayerParams(object):
    '''
    classdocs
    '''
    yaml_tag = u'!LayerParams'
    
    def __init__(self, inputDim, outputDim):
        '''
        Constructor
        '''        
        self.LayerClass = None
        self._inputDim = inputDim
        self._outputDim = outputDim
        self._name = None
        self._inputs = []
        self.initValues = None
        
    def initFromConfig(self,cfg,sectionKey):    
        self._name = sectionKey
        
        self._inputs = []
        layerInputNames = readCfgStrListParam(cfg, sectionKey, 'inputs', [])
        for iln in layerInputNames:
            m = re.match("input\[(\d+)\]",iln)
            if m:
                inputVarNum = int(m.group(1))
                self._inputs.append(inputVarNum)
            else:
                self._inputs.append(iln)
            
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self,value):
        self._name = value
    
    @property
    def inputs(self):
        return self._inputs    

    @inputs.setter
    def inputs(self,value):
        assert isinstance(value,list), "inputs must be a list"
        assert all([(isinstance(v,int) or isinstance(v,basestring)) for v in value]), "inputs must be a list of int or str"
        self._inputs = value      
                 
    @property
    def outputDim(self):
        return self._outputDim 

    @outputDim.setter
    def outputDim(self,value):
        self._outputDim = value;
        self.update()

    @property
    def inputDim(self):
        return self._inputDim 

    @inputDim.setter
    def inputDim(self,value):
        self._inputDim = value
        self.update()
        
                
    def update(self):
        '''
        Default. Override in derived
        '''
        #self._outputDim
        #self._inputDim
        pass
        
        
    def checkInputDim(self,expectSingleInput=False,expectMultipleInputs=False):

        if expectMultipleInputs and expectSingleInput:
            raise ValueError("Coding error: Cannot expect single and multiple inputs at the same time")
        
        if self._inputDim is None:
            raise ValueError("inputDim empty")
        if not isinstance(self._inputDim,collections.Iterable):
            raise ValueError("inputDim is neither a list nor a tuple ...")
            
        if expectSingleInput and isinstance(self._inputDim[0],collections.Iterable):
            raise ValueError("Can only deal with single input, but inputDim is {}".format(self._inputDim))

        if expectMultipleInputs and not isinstance(self._inputDim[0],collections.Iterable):
            raise ValueError("Layer expects more than one input, but inputDim is {}".format(self._inputDim))

        
    def checkOutputDim(self):
        if any([d < 0 for d in self._outputDim]):
            raise ValueError("Negative output dimensionality {}".format(self._outputDim))
             

    def __getstate__(self):
        state = dict()
        state['LayerClass'] =  self.LayerClass
        state['inputDim'] = self._inputDim
        state['outputDim'] = self._outputDim
        state['name'] = self._name
        state['inputs'] = self._inputs
        state['initValues'] = self.initValues
        return state
    
    def __setstate__(self,state):
        self.LayerClass = state['LayerClass']
        self._inputDim = state['inputDim'] 
        self._outputDim = state['outputDim'] 
        self._name = state['name']
        self._inputs = state['inputs'] 
        self.initValues = state['initValues'] 


class Layer(object):
    
    def __init__(self):
        pass
    
    def saveValues(self,fileName):
        '''
        Actually just have to store parameters and put the file where we stored them in the cfgParams
          that way we can be reconstructed by creating a new instance and just running through init
          
        -> thus in layers without parameters (that cannot be reconstructed from the cfgParams), no need to save anything
        '''
