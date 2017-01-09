'''
Created on 05.05.2014

@author: pwohlhart
'''

import numpy
import theano

import cPickle

from data import transformations
import data.importers
from data.basetypes import PairIdx, NamedImgSequence, PairStacks
from data.importers import LinemodTrainDataImporter



#PairStacks = namedtuple('PairStacks',['x0','x1','y'])  ... the named tuple thing is immutable, hm :-|

class Dataset(object):
    '''
    ----
    '''

    def __init__(self, imgSeqs=None):
        '''
        Constructor
        
        _imgSeqs: NamedImgSequence
        _imgStacks: dict of ndarrays with the image data (not used anymore)
        '''
        if imgSeqs is None:
            self._imgSeqs = []
        else:
            self._imgSeqs = imgSeqs
        self._imgStacks = {}
        
    @property  
    def imgSeqs(self):
        return self._imgSeqs
    
    @imgSeqs.setter
    def imgSeqs(self, value):
        self._imgSeqs = value
        self._imgStacks = {}
        
    def imgStack(self,seqName):
        floatX = theano.config.floatX  # @UndefinedVariable
        imgSeq = None
        for seq in self._imgSeqs:
            if seq.name == seqName:
                imgSeq = seq
                break
        if imgSeq is None:
            return []
        
        if not self._imgStacks.has_key(seqName):
            # compute the stack from the sequence
            numImgs = len(imgSeq.data)
            data0 = numpy.asarray(imgSeq.data[0].img, floatX) 
            h,w,c = data0.shape
            imgStack = numpy.zeros((numImgs,c,h,w),dtype=floatX)  # num_imgs,stack_size,rows,cols 
            for i in range(numImgs):
                imgStack[i] = numpy.rollaxis( numpy.asarray(imgSeq.data[i].img,floatX) ,2) # take i-th img and shuffle HxWxRGB -> RGBxHxW
            #st = numpy.array;  # ?
            imgStack /= 255.
            self._imgStacks[seqName] = imgStack  
            
        return self._imgStacks[seqName]
    

    def createRandomPairsIdx(self,nSame,nDiff):
        '''
        we have a list of image sequences.
        select random "same" and "different" pairs
        and return them in PairStack(imgStack1,imgStack2,labels) form 

        :type nSame: int
        :param nSame: number of "same" pairs, per sequence
        
        :type nDiff: int
        :param nDiff: number of "different" pairs
        '''
        
        sameSeq0Idx = []
        sameSeq1Idx = []
        labels = []
        
        numSeqs = len(self._imgSeqs)
        
        # same pairs
        for i in range(numSeqs):
            seq = self._imgSeqs[i]
            data = seq.data
            idx0 = numpy.random.randint(len(data),size=nSame) # TODO: if nSame >= len(seq) we could make sure that every item is at least taken once by taking permutations
            idx1 = numpy.random.randint(len(data)-1,size=nSame)
            idx1[idx1>=idx0] += 1   # assert: all(idx0 != idx1)
            sameSeq0Idx.extend(zip([i]*nSame,idx0))
            sameSeq1Idx.extend(zip([i]*nSame,idx1))
            
            #for j in range(nSame):
            #    seq0Idx.append((i,idx0[j]))
            #    seq1Idx.append((i,idx1[j]))
            #    labels.append(1)
        
        labels = [1]*(nSame*numSeqs)
    
        # different pairs
        
        #  code for permutations: [x for x in itertools.permutations(range(numSeqs),2)]
        #    but then we would have to replicate and cut them such as to have exactly nDiff
        
        diffSeq0Idx = []
        diffSeq1Idx = []        
        #seqNames = self._imgSeqs.keys()
        for seqIdx in range(numSeqs):
            seq = self._imgSeqs[seqIdx]
            #seq = self._imgSeqs[seqNames[seqIdx]]
            idx0 = numpy.random.randint(len(seq.data),size=nDiff)
            for j in range(nDiff):
                otherSeqIdx = numpy.random.randint(numSeqs-1)
                if otherSeqIdx >= seqIdx:
                    otherSeqIdx += 1
                otherSeq = self._imgSeqs[otherSeqIdx]
                idx1 = numpy.random.randint(len(otherSeq.data))
                diffSeq0Idx.append((seqIdx,idx0[j]))
                diffSeq1Idx.append((otherSeqIdx,idx1))
                #labels.append(0)
        labels.extend([0]*(nDiff*numSeqs))
        
        # shuffle indizes such that there is always one positive and one negative sample
        assert nSame == nDiff # for now. otherwise the shuffling does not work like this
        seq0Idx = [val for pair in zip(sameSeq0Idx, diffSeq0Idx) for val in pair]
        seq1Idx = [val for pair in zip(sameSeq1Idx, diffSeq1Idx) for val in pair]
        labels = [val for pair in zip([1]*(nSame*numSeqs), [0]*(nSame*nDiff)) for val in pair]
         
        return PairIdx(seq0Idx,seq1Idx,labels)
      
      
    def getPairImgStacks(self,pairIdx):
        '''
        Takes pairs of indizes into _imgSeq
        and returns stacks in form of numpy arrays ()  
        
        :param pairIdx: PairIdx into _imgSeq
        
        :return : PairStacks
        '''
        floatX = theano.config.floatX  # @UndefinedVariable
        
        img0 = self._imgSeqs[0].data[0].img
        data0 = numpy.asarray(img0, floatX) 
        h,w,c = data0.shape
        
        numPairs = len(pairIdx.idx1)
        
        imgStack0 = numpy.zeros((numPairs,c,h,w),dtype=floatX);  # num_imgs,stack_size,rows,cols
        imgStack1 = numpy.zeros((numPairs,c,h,w),dtype=floatX);  # num_imgs,stack_size,rows,cols
        #labels = numpy.zeros(numPairs)
        
        # compute the stack from the sequence
        for i in range(numPairs):
            idx0 = pairIdx.idx0[i]
            imgStack0[i] = numpy.rollaxis( numpy.asarray(self._imgSeqs[idx0[0]].data[idx0[1]].img,floatX) ,2) # take i-th img and shuffle HxWxRGB -> RGBxHxW
            idx1 = pairIdx.idx1[i]
            imgStack1[i] = numpy.rollaxis( numpy.asarray(self._imgSeqs[idx1[0]].data[idx1[1]].img,floatX) ,2) # take i-th img and shuffle HxWxRGB -> RGBxHxW
        
        imgStack0 /= 255
        imgStack1 /= 255
        labels = numpy.asarray(pairIdx.labels, dtype=floatX)
            
        return PairStacks(imgStack0,imgStack1,labels)
    
    def jitterImgSeqs(self,N,maxRot=30.0,maxTranslate=2.0,maxScale=0.1):
        imj = transformations.ImageJitterer()
        seqs = []
        for imgSeq in self._imgSeqs:
            jitteredImgs = imj.getJitteredImgSeq(imgSeq.data,N,maxRot=maxRot,maxTranslate=maxTranslate,maxScale=maxScale)
            seqs.append(NamedImgSequence(imgSeq.name, jitteredImgs))
        self._imgSeqs = seqs
    
    
    
class LinemodDataset(Dataset):
    
    def __init__(self,basepath=None):
        '''
        constructor
        '''
        if basepath is None:
            basepath = '../data/linemod/'
        
        self.lmi = data.importers.LinemodImporter(basepath)  
        
    def loadFromFiles(self,objNames=None,zRotInv=None):
        
        if objNames is None:
            objNames = self.lmi.getObjectNames()
            zRotInv = [0]*len(objNames) # we dont know, so we assume not
        self._imgSeqs = self.lmi.loadSequences(objNames,zRotInv) 
         
        # TODO ........
        #assert False, "this needs to be fully implemented before using"
        
    def loadFromPkl(self,filename):
        
        with open(filename,'rb') as f:
            self._imgSeqs = cPickle.load(f)
        
    def saveToPkl(self,filename):
        
        with open(filename,'wb') as f:
            cPickle.dump(self._imgSeqs,f,protocol=cPickle.HIGHEST_PROTOCOL)
            

# class LinemodTmplDataset(Dataset):        
#         
#     def __init__(self,basepath=None):
#         '''
#         constructor
#         '''
#         if basepath is None:
#             basepath = '../data/linemod/'
#         
#         self.lmi = data.importers.LinemodImporter(basepath)  
#         self.lmti = LinemodTrainDataImporter()
#                 
#     def loadFromFiles(self,objNames=None,inputMode=0,inputSize=64):
#         
#         if objNames is None:
#             objNames = self.lmi.getObjectNames()
#             
#         zRotInv = (0,1,0,0) 
#         
#         imgNumsTemplates = numpy.arange(301)
#         
#         # TEMPLATES
#         imgSeqs = []
#         for i in range(len(objNames)):
#             seqN = objNames[i]
#             zri = zRotInv[i]
#             imgSeq = self.lmti.loadSequence(objName=seqN,imgNums=imgNumsTemplates,cropSize=20.0,targetSize=targetSize,zRotInv=zri)
#             imgSeqs.append(imgSeq)
#         
#         self._imgSeqs = imgSeqs
        
            
    