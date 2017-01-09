'''
Created on 29.04.2014

@author: pwohlhart
'''

import cPickle
import gzip
import numpy as np

class LinemodObjStore(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        
    def saveSequenceData(self,data,filename):
        '''
        pickle data loaded with Importer  
        '''
        
        f = gzip.open(filename, 'wb')
        cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        
    def loadSequenceData(self,filename):
        '''
        pickle data loaded with Importer  
        '''
        
        f = gzip.open(filename, 'rb')
        data = cPickle.load(f)
        f.close()        
        return data
    
    def saveSequenceAsArray(self,seq,filename,dtype=np.float32):
        '''
        data is list of tuples (img,dpt,rot,tra)
        '''
        
        print("converting to 4D arrays")
        # Color Image
        imgStack = self.cvtSequenceImgsToArray(seq,dtype)
        
        # Depth Image
        dptStack = self.cvtSequenceDptsToArray(seq, dtype)
        
        print("saving")
        #f = gzip.open(filename, 'wb')
        f = open(filename, 'wb')
        cPickle.dump((imgStack,dptStack), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()        
        
        
    def loadSequenceAsArray(self,filename):
        
        f = open(filename, 'rb')
        data = cPickle.load(f)
        f.close()        
        return data
    
    
    def cvtSequenceImgsToArray(self,seq,dtype=np.float32):
    
        numImgs = len(seq)
        h,w,c = seq[0][0].shape
        imgStack = np.zeros((numImgs,c,h,w),dtype=dtype);  # num_imgs,stack_size,rows,cols 
        
        for i in range(0,numImgs):
            imgStack[i] = np.rollaxis(seq[i][0],2) # take i-th img and shuffle HxWxRGB -> RGBxHxW
                    
        return imgStack
    
    def cvtSequenceDptsToArray(self,seq,dtype=np.float32):
    
        numImgs = len(seq)
        h,w = seq[0][1].shape
        dptStack = np.zeros((numImgs,1,h,w),dtype=dtype);  # num_imgs,stack_size,rows,cols
        
        for i in range(0,numImgs):
            dptStack[i] = seq[i][1][np.newaxis,:,:] # take i-th dpt and make it HxW -> 1xHxW
                                
        return dptStack    
    