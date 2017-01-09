'''
Created on 30.04.2014

@author: pwohlhart
'''

import numpy as np
from PIL import Image
import data.basetypes
import cv2
import numpy
import heapq


def getTransformationMatrix(center,rot,trans,scale):
    ca = np.cos(rot)
    sa = np.sin(rot)
    sc = scale
    cx = center[0]
    cy = center[1]
    tx = trans[0]
    ty = trans[1]
    t = np.array([ca*sc ,-sa*sc, sc*(ca*(-tx-cx) + sa*( cy + ty)) + cx, 
                  sa*sc,  ca*sc, sc*(ca*(-ty-cy) + sa*(-tx - cx)) + cy])
    return t

class ImageJitterer(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
    def getJitteredParams(self,num,center=(0.0,0.0),maxRot=(-5.0,5.0),maxTranslate=(-2.0,2.0),maxScale=(-0.1,0.1)):
        
        if not (type(maxRot) is tuple):
            maxRot = (-maxRot,maxRot)
        if not (type(maxTranslate) is tuple):
            maxTranslate = (-maxTranslate,maxTranslate)
        if not (type(maxScale) is tuple):
            maxScale = (-maxScale,maxScale)
        
        alphas = np.random.rand(num)*(maxRot[1]-maxRot[0]) + maxRot[0]
        alphas = np.deg2rad(alphas)
        ca = np.cos(alphas)
        sa = np.sin(alphas)
        
        tx = np.random.rand(num)*(maxTranslate[1]-maxTranslate[0]) + maxTranslate[0] 
        ty = np.random.rand(num)*(maxTranslate[1]-maxTranslate[0]) + maxTranslate[0]
        
        sc = 2**-(np.random.rand(num)*(maxScale[1]-maxScale[0]) + maxScale[0])
        
        cx = center[0]
        cy = center[1]
        
        transformationMats = []
        for i in range(num):
            #-- translate only
            #t = np.array([1,0,cx,0,1,cy])
            #-- translate, rotate
            #t = np.array([ca[i],-sa[i],ca[i]*cx-cy*sa[i], sa[i],ca[i],ca[i]*cy+cx*sa[i]])
            # full: translate to center, jitter rotation, jitter scale, jitter translation, translate back
            #t = getTransformationMatrix(alphas[i],sc[i],)
            t = np.array([ca[i]*sc[i] ,-sa[i]*sc[i], sc[i]*(ca[i]*(-tx[i]-cx) + sa[i]*( cy + ty[i])) + cx, 
                          sa[i]*sc[i],  ca[i]*sc[i], sc[i]*(ca[i]*(-ty[i]-cy) + sa[i]*(-tx[i] - cx)) + cy])
            transformationMats.append(t)
            
        return transformationMats
        
        
    def getJitteredImgs(self,img,num,maxRot=(-5.0,5.0),maxTranslate=(-2.0,2.0),maxScale=(-0.1,0.1)):
        
        ''' ///////
        if not (type(maxRot) is tuple):
            maxRot = (-maxRot,maxRot)
        if not (type(maxTranslate) is tuple):
            maxTranslate = (-maxTranslate,maxTranslate)
        if not (type(maxScale) is tuple):
            maxScale = (-maxScale,maxScale)
        
        alphas = np.random.rand(num)*(maxRot[1]-maxRot[0]) + maxRot[0]
        alphas = np.deg2rad(alphas)
        ca = np.cos(alphas)
        sa = np.sin(alphas)
        
        tx = np.random.rand(num)*(maxTranslate[1]-maxTranslate[0]) + maxTranslate[0] 
        ty = np.random.rand(num)*(maxTranslate[1]-maxTranslate[0]) + maxTranslate[0]
        
        sc = 2**-(np.random.rand(num)*(maxScale[1]-maxScale[0]) + maxScale[0])
        
        transformedSize = img.size 
        cx = img.size[0]/2
        cy = img.size[1]/2
        
        imgs = []
        for i in range(num):
            # compile this function using sympy:
            #   tx = Symbol('tx')   
            #
            #-- translate only
            #t = np.array([1,0,cx,0,1,cy])
            #-- translate, rotate
            #t = np.array([ca[i],-sa[i],ca[i]*cx-cy*sa[i], sa[i],ca[i],ca[i]*cy+cx*sa[i]])
            # full: translate to center, jitter rotation, jitter scale, jitter translation, translate back
            t = np.array([ca[i]*sc[i] ,-sa[i]*sc[i], sc[i]*(ca[i]*(-tx[i]-cx) + sa[i]*( cy + ty[i])) + cx, 
                          sa[i]*sc[i],  ca[i]*sc[i], sc[i]*(ca[i]*(-ty[i]-cy) + sa[i]*(-tx[i] - cx)) + cy])
            imgT = img.transform(transformedSize,Image.AFFINE,t,Image.BILINEAR)
            
            # TODO: jitter colors / brightness / contrast / ...
            #       add noise
            
            imgs.append(imgT)
            
        return imgs
        '''
        
        cx = img.size[0]/2
        cy = img.size[1]/2
        transformedSize = img.size
        
        tMats = self.getJitteredParams(center=(cx,cy),num=num,maxRot=maxRot,maxTranslate=maxTranslate,maxScale=maxScale)
        imgs = []
        for i in range(len(tMats)):
            t = tMats[i]
            imgT = img.transform(transformedSize,Image.AFFINE,t,Image.BILINEAR)

            # TODO: jitter colors / brightness / contrast / ...
            #       add noise
            
            imgs.append(imgT)
            
        return imgs
        
    def getJitteredImgSeq(self,imgSeq,num,maxRot=(-5.0,5.0),maxTranslate=(-2.0,2.0),maxScale=(-0.1,0.1)):
        '''
        Take every img in the sequence (ie. list of Frames) and jitter it
        return a list of all jittered 
        
        :param imgSeq: list of Frames
        '''
        
        seq = []
        for frame in imgSeq:
            imgs = self.getJitteredImgs(frame.img,num,maxRot,maxTranslate,maxScale)
            for i in range(num):
                seq.append(data.basetypes.Frame(imgs[i],frame.dpt,frame.pose,frame.className,frame.filename))
        return seq
    
    
    
    
def dptInpaint(dpt,mask=None):
    
    if mask is None:
        mask = (dpt==0).astype(numpy.uint8)
    
    r = 1
   
    h,w = dpt.shape

    #add border
    dpt = cv2.copyMakeBorder(dpt,r,r,r,r,borderType=cv2.BORDER_CONSTANT,value=0)
    mask = cv2.copyMakeBorder(mask,r,r,r,r,borderType=cv2.BORDER_CONSTANT,value=2.)
    
    #intMask = cv2.integral(mask)
    
    # TODO: do some im2col like stuff
    #  like in http://stackoverflow.com/questions/10896841/find-a-3x3-sliding-window-over-an-image
    
    res = numpy.copy(dpt)
    msk = numpy.copy(mask)
    while numpy.any(mask==1):
        mod = False
        #cv2.imshow("mask",mask*127)    
        #cv2.imshow("res",res/400.)
        #cv2.waitKey()
        for y in numpy.arange(r,h+r):
            for x in numpy.arange(r,w+r):
                if mask[y,x] > 0:
                    m = mask[y-r:y+r+1,x-r:x+r+1] == 0
                    #print("nsm {}".format(numpy.sum(m)))
                    p = dpt[y-r:y+r+1,x-r:x+r+1][m]   
                    #print("p.size {}, {}".format(p.size,p))         
                    if p.size > 0:
                        numpy.sort(p)
                        #print("b {}".format(res[y,x]))
                        res[y,x] = p[p.size/2]
                        #print("a {}".format(res[y,x]))
                        msk[y,x] = 0
                        mod = True
        if mod:
            dpt = numpy.copy(res)
            mask = numpy.copy(msk)
        else:
            break
            
    res = res[r:h+r,r:w+r]
                
    #dp = im2col(dpt,(3,3))
    #mp = im2col(mask,(3,3))
    #res = numpy.zeros((dp.shape[1],),dtype=dpt.dtype)
    #for i in range(dp.shape[1]):
    #    v = dp[:,i][mp[:,i]>0]
    #    if v.size > 0:
    #        numpy.sort(v)
    #        res[i] = v[v.size/2]
   
    #print("mn/mx {},{}".format(numpy.min(dpt),numpy.max(dpt)))
    #print("{},{}, {}".format(h,w,res.shape))
    #cv2.imshow("dpt",dpt/400.)    
    #cv2.imshow("res",res/400.)
    #cv2.waitKey()

    #return res.reshape(h,w)            
    return res
            

def dptInpaintV2(dpt,mask=None):
    
    if mask is None:
        mask = (dpt==0).astype(numpy.uint8)
    
    r = 1
    s = 2*r+1
   
    h,w = dpt.shape

    #add border
    dpt = cv2.copyMakeBorder(dpt,r,r,r,r,borderType=cv2.BORDER_CONSTANT,value=0)
    mask = cv2.copyMakeBorder(mask,r,r,r,r,borderType=cv2.BORDER_CONSTANT,value=2.)
    
    #intMask = cv2.integral(mask)
    
    # TODO: do some im2col like stuff
    #  like in http://stackoverflow.com/questions/10896841/find-a-3x3-sliding-window-over-an-image
    
    res = numpy.copy(dpt)
    msk = numpy.copy(mask)
    while numpy.any(mask==1):
        mod = False
        #cv2.imshow("mask",mask*127)    
        #cv2.imshow("res",res/400.)
        #cv2.waitKey()
        for y in numpy.arange(r,h+r):
            for x in numpy.arange(r,w+r):
                if mask[y,x] > 0:
                    m = mask[y-r:y+r+1,x-r:x+r+1].flatten()
                    p = dpt[y-r:y+r+1,x-r:x+r+1].flatten()
                    
                    minHeap = []  # keeps smallest of the larger numbers at minHeap[0]
                    maxHeap = []  # keeps largest of the smaller number at maxHeap[0]. actually it also has the smallest value in front, so we insert inverted values (*-1)
                    for i in xrange(s):
                        if m[i] == 0:
                            pi = p[i]
                            if not maxHeap:
                                if not minHeap:     
                                    minHeap = [pi]   # both empty, add left
                                else:
                                    if pi > minHeap[0]:  # there is something in the minHeap that is smaller -> swap the two
                                        maxHeap = [-minHeap[0]]
                                        minHeap[0] = pi
                                    else:
                                        maxHeap = [-pi]
                            else:   # both heaps have at least one element
                                if pi < -maxHeap[0]:
                                    heapq.heappush(maxHeap,-pi)
                                else:
                                    heapq.heappush(minHeap,pi)
                                    
                                # balance heaps
                                if len(maxHeap)>len(minHeap)+1:
                                    # move one from maxHeap to minHeap
                                    heapq.heappush(minHeap,-heapq.heappop(maxHeap))
                                elif len(minHeap)>len(maxHeap)+1:
                                    heapq.heappush(maxHeap,-heapq.heappop(minHeap))
                            
                        if minHeap:                    
                            if not maxHeap:
                                pi = minHeap[0]
                            else:
                                if len(maxHeap) == len(minHeap):
                                    pi = (minHeap[0] - maxHeap[0])/2.
                                elif len(maxHeap) > len(minHeap):
                                    pi = -maxHeap[0]
                                else:
                                    pi = minHeap[0]
                            
                            res[y,x] = pi
                            #print("a {}".format(res[y,x]))
                            msk[y,x] = 0
                            mod = True
                            
        if mod:
            dpt = numpy.copy(res)
            mask = numpy.copy(msk)
        else:
            break
            
    res = res[r:h+r,r:w+r]
                
    #dp = im2col(dpt,(3,3))
    #mp = im2col(mask,(3,3))
    #res = numpy.zeros((dp.shape[1],),dtype=dpt.dtype)
    #for i in range(dp.shape[1]):
    #    v = dp[:,i][mp[:,i]>0]
    #    if v.size > 0:
    #        numpy.sort(v)
    #        res[i] = v[v.size/2]
   
    #print("mn/mx {},{}".format(numpy.min(dpt),numpy.max(dpt)))
    #print("{},{}, {}".format(h,w,res.shape))
    #cv2.imshow("dpt",dpt/400.)    
    #cv2.imshow("res",res/400.)
    #cv2.waitKey()

    #return res.reshape(h,w)            
    return res
            
def im2col(Im, block, style='sliding'):
    """block = (patchsize, patchsize)
        first do sliding
    """
    bx, by = block
    Imx, Imy = Im.shape
    Imcol = []
    for j in range(0, Imy):
        for i in range(0, Imx):
            if (i+bx <= Imx) and (j+by <= Imy):
                Imcol.append(Im[i:i+bx, j:j+by].T.reshape(bx*by))
            else:
                break
    return numpy.asarray(Imcol).T            
    