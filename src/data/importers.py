'''
Created on 28.04.2014

@author: pwohlhart
'''

#from __future__ import with_statement # This isn't required in Python 2.6

import glob
import os
import cv2
import numpy
import theano

from PIL import Image
from PIL import ImageDraw
import progressbar as pb
from matplotlib import cm
from numpy import math
from data.basetypes import Frame, NamedImgSequence, LinemodObjPose, LinemodTestFrame, Patch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # @UnresolvedImport
from data.camera import LinemodCam, BlenderCam


class LinemodImporter(object):
    '''
    provide functionality to load data from the Linemod dataset  
    '''


    def __init__(self,basepath):
        '''
        Constructor
        '''
        self.basepath = basepath
        
    def loadDepthMap(self,filename=None,objName=None,imgNum=None):
        '''
        Read a kinect depth-map as stored with the linemod dataset 
        '''
        
        if filename is None:
            filename = '{basepath}{objname}/data/depth{num}.dpt'.format(basepath=self.basepath,objname=objName,num=imgNum)
        
        _,ext = os.path.splitext(filename)
        if ext == ".dpt":
            with open(filename) as f:
                h,w = numpy.fromfile(f,dtype=numpy.uint32,count=2)
                #print('w {}, h {}'.format(w,h))
                
                data = numpy.fromfile(f,dtype=numpy.uint16,count=w*h)
                data = data.reshape((h,w))
                
                data = data.astype(numpy.float32)/10.  # make it cm, as everything else is in cm now
                
        elif ext == ".png":
            
            data = cv2.imread(filename,cv2.CV_LOAD_IMAGE_UNCHANGED).astype(numpy.float32) / 10.
            
        else:
            raise ValueError("Unkown depth image file format '{}'".format(ext))
            
        return data
        
        
    def loadImage(self,filename=None,objName=None,imgNum=None):
        
        if filename is None:
            filename = '{basepath}{objname}/data/depth{num}.dpt'.format(basepath=self.basepath,objname=objName,num=imgNum)

        #im = Image.open(filename)
        #data = np.asarray(im, np.uint8)
        
        data = cv2.imread(filename) / 255. - 0.5
        return data
    
    
    def loadMatFromTxt(self,filename):
        return numpy.loadtxt(filename,skiprows=1)
        
        
    def loadSequence(self,seqName,zRotInv=0,inputMode=2,cropAtGtPos=False,cropSize=None,targetSize=None,imgNums=None):
        
        objdir = '{}{}/data/'.format(self.basepath, seqName) 
        s = '{}color*.jpg'.format(objdir,self.basepath, seqName)
        
        imgFileNames = glob.glob(s)
        imgFileNames = sorted(imgFileNames)

        assert len(imgFileNames) > 0, "No images under '{}'".format(s)
                
        # tmp
        #imgFileNames = imgFileNames[0:100]
        #imgFileNames = imgFileNames[0:19]
        #imgFileNames = imgFileNames[0:800]
        if imgNums is not None:
            imgFileNames = [imgFileNames[i] for i in imgNums]
        
        print("inputMode {}".format(inputMode))
        
        txt = 'Loading {}'.format(seqName)
        pbar = pb.ProgressBar(maxval=len(imgFileNames),widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()
        
        data = []
        for i in range(0,len(imgFileNames)):
            imgFileName = imgFileNames[i]
            d,fn = os.path.split(imgFileName)  # @UnusedVariable
            numStr = fn[5:-4]
            #dptFileName = '{}depth{}.dpt'.format(objdir,numStr)
            #dptFileName = '{}inp/depth{}.dpt'.format(objdir,numStr)  # inpainted
            dptFileName = '{}inp/depth{}.png'.format(objdir,numStr)  # inpainted
            rotFileName = '{}rot{}.rot'.format(objdir,numStr)
            traFileName = '{}tra{}.tra'.format(objdir,numStr)
            
            if (inputMode > 0):
                img = self.loadImage(imgFileName)
            else:
                img = None
            if (inputMode != 1):
                dpt = self.loadDepthMap(dptFileName)
            else:
                dpt = None    
            mask = None # or numpy.ones with the size of img?
            rot = self.loadMatFromTxt(rotFileName)
            tra = self.loadMatFromTxt(traFileName)
        
            #data.append({img:img,dpt:dpt,rot:rot,tra:tra})
            pose = LinemodObjPose(rot=rot,tra=tra,zRotInv=zRotInv)
            frame = LinemodTestFrame(img=img,dpt=dpt,mask=mask,pose=pose,className=seqName,filename=dptFileName)
            if cropAtGtPos:
                patch = self.cropPatchAtGTPos(frame,cropSize,targetSize,inputMode)
                #print("patch mn/mx {},{}".format(numpy.min(patch),numpy.max(patch)))
                data.append( patch )
                # keep only the first few full frames for debugging
                if i > 10:
                    frame.img = None
                    frame.dpt = None
                    frame.mask = None
            else:
                data.append( frame )
            pbar.update(i)
                        
        return NamedImgSequence(seqName,data)


    def cropPatchAtGTPos(self,frame,cropSize,targetSize,inputMode,normalizeDepth=True,debug=False):
        
        lcam = LinemodCam()
        
        rcp = frame.pose.relCamPos
        mat = frame.pose.getMatrix()
        worldPt = numpy.array([0,0,-5,1.])
        camPt = numpy.dot(mat,worldPt)  # center pt in camera space
        screenPt = lcam.worldToScreen(camPt)

        v = camPt[0:3]   # vector cam center - object center in cam space
        
        if (rcp[0] == 0) and (rcp[1] == 0):  # looking down from exactly on top -> cannot do the cross product with the z axis
            u = numpy.array([0.,1.,0.])
            r = numpy.array([1.,0.,0.])
            #print "---------------------------  holla die waldfee --------------------------------------------"
        else:
            ignoreInplaneRotation = False
            if ignoreInplaneRotation:
                # up vector of rectangle to crop in the world = x-axis X v = (1,0,0) X (vx,vy,vz) = (0,-vz,vy)    (or really it is down (depending on the coord system))
                u = numpy.array([0.,-v[2],v[1]])
                # right vector of rectangle to crop = u X v = (wy^2+cropDepth^2,-wx*wy,-wx*cropDepth) = (vy^2+vz^2, vx*vy, -vy*vz) 
                r = numpy.array([v[1]*v[1]+v[2]*v[2],-v[0]*v[1],-v[0]*v[2]])
            else:
                upV = (numpy.dot(mat,numpy.array([0,0,-6,1.])) - camPt)[0:3]
                r = numpy.cross(v,upV)
                u = numpy.cross(r,v)
            
#         if debug:
#             print("m {}".format(mat))
#             print("w {}".format(worldPt))
#             print("c {}".format(camPt))
#             print("s {}".format(screenPt))
#             print("v {}".format(v))
#             print("u {}".format(u))
#             print("r {}".format(r))
#             print("rcp {}".format(rcp))            

        # normalize, resize
        u = u * (cropSize/2) / numpy.linalg.norm(u)
        r = r * (cropSize/2) / numpy.linalg.norm(r)
        
        # world coord of corner points
        w1 = v + u - r
        w2 = v - u - r
        w3 = v + u + r
        w4 = v - u + r
        
        # coords of corner points on image plane
        worldRect = numpy.concatenate((w1.reshape((1,3)),w2.reshape((1,3)),w3.reshape((1,3)),w4.reshape((1,3))),axis=0)
        sourceRect = lcam.worldToScreen(worldRect).astype(numpy.float32)
              
        # target pts 
        targetRect = numpy.array([[0,0],[0,1],[1,0],[1,1]]).astype(numpy.float32) * targetSize
        M = cv2.getPerspectiveTransform(sourceRect,targetRect)
        
        mask = None
               
        ts = (targetSize,targetSize)
        if inputMode != 1:
            # frame.dpt is data in float32 measuring depth in cm
            dpt = numpy.copy(frame.dpt)
            
            # fix holes
#             fixHolesMode = -1  
#             if fixHolesMode == 0:
#                 dpt[dpt==0] = numpy.max(dpt)+1.
#             elif fixHolesMode == 1:
#                 inpaintMask = (dpt == 0).astype(numpy.uint8)
#                 dpt = dptInpaint(dpt,inpaintMask)
            
            dptPatch = cv2.warpPerspective(dpt,M,ts)
            
            if frame.mask is not None:
                mask = cv2.warpPerspective(frame.mask,M,ts)
            
            if normalizeDepth:
                dHigh = 20. 
                dLow = -20.
                dptPatch = dptPatch - camPt[2]  # subtract depth of extraction point (center values on it)
                dptPatch = numpy.maximum(numpy.minimum(dptPatch,dHigh),dLow)
                dptPatch /= numpy.maximum(abs(dHigh),abs(dLow)) # -> / 20.  => -1..1
        else:
            dptPatch = None
            
        if inputMode != 0:
            if frame.img.dtype == numpy.float32:
                imgPatch = cv2.warpPerspective(frame.img+0.5,M,ts)-0.5  # do we need this +/- 0.5, or can the values be negative in the process?
            else:
                imgPatch = cv2.warpPerspective(frame.img,M,ts) 
        else:
            imgPatch = None  

        return Patch(img=imgPatch,dpt=dptPatch,mask=mask,frame=frame,cropArea=sourceRect)
    

    
    def getObjectNames(self):
        dirlist = [d for d in os.listdir(self.basepath) if os.path.isdir(os.path.join(self.basepath,d))]
        objNames = [d for d in dirlist if os.path.isfile(os.path.join(self.basepath,d,'data','color0.jpg'))] 
        return sorted(objNames)
        
        
    def loadSequences(self,seqNames,zRotInv=None,inputMode=2):
        '''
        :returns: list of NamedImgSequence
        '''
        data = []
        
        #for seqName in seqNames:
        for i in xrange(len(seqNames)):
            seqData = self.loadSequence(seqNames[i],zRotInv=zRotInv[i],inputMode=inputMode)
            data.append(seqData)
            
        return data

        
    
    
class DummyImporter(object):
    
    def __init__(self):
        pass
    
    def loadSequence(self,objName):
        '''
        :returns: NamedImgSequence
        '''
        
        w = 64
        img = Image.new('RGB', size=(w,w), color=(0,0,0))
        draw = ImageDraw.Draw(img)
        
        b = math.floor(w/5)
        
        if objName == 'circle':
            draw.ellipse([(b,b),(w-b,w-b)], fill=(255,0,0))
            
        elif objName == 'square':
            draw.rectangle([(b,b),(w-b,w-b)], fill=(255,0,0))

        elif objName == 'triangle':
            draw.polygon([(w/2,b),(b,w-b),(w-b,w-b)], fill=(255,0,0))

        del draw

        data = []
        data.append(Frame(img,None,None,None,objName))
        return NamedImgSequence(objName,data)
            
    def getObjectNames(self):
        return ['circle', 'square', 'triangle']
    
    
    
class ToySequencesImporter(object):
    
    def __init__(self):
        pass
    
    def loadSequence(self,objName):
        '''
        :returns: NamedImgSequence
        '''
        
        w = 64
        img = Image.new('RGB', size=(w,w), color=(0,0,0))
        draw = ImageDraw.Draw(img)
        
        b = math.floor(w/5)
        
        if objName == 'circle':
            draw.ellipse([(b,b),(w-b,w-b)], fill=(255,0,0))
            
        elif objName == 'square':
            draw.rectangle([(b,b),(w-b,w-b)], fill=(255,0,0))

        elif objName == 'triangle':
            draw.polygon([(w/2,b),(b,w-b),(w-b,w-b)], fill=(255,0,0))

        del draw

        data = []
        data.append(Frame(img,None,None,None,objName,""))
        return NamedImgSequence(objName,data)
            
    def getObjectNames(self):
        return ['circle', 'square', 'triangle']
    


        
class LinemodTrainDataImporter(object):
    '''
    provide functionality to load Linemod training data, created with Blender  
    '''

    def __init__(self,basePath=None):
        
        if basePath == None:
            self.basePath = "/home/wohlhart/work/data/linemod/"
#            self.basePath = "/home/wohlhart/work/data/linemod/blender_experiments/tmp_renders/"
        else:
            self.basePath = basePath
            
        pass
    
    def loadSequence(self,objName,imgNums=None,cropSize=None,targetSize=None,zRotInv=0):
        
        floatX = theano.config.floatX  # @UndefinedVariable
        
        if cropSize is None:
            cropSize = 20.0 # 28.
        if targetSize is None:
            targetSize = 64
        
        seqData = []
        
        #cam = BlenderCam()
        
        camPositionsElAz = numpy.loadtxt(self.basePath + 'camPositionsElAz.txt')
        camOrientations = numpy.loadtxt(self.basePath + 'camOrientations.txt')
        if camOrientations.ndim < 2:
            camOrientations = camOrientations.reshape((len(camOrientations)/2,2))
         

        numCams = len(camPositionsElAz)
        camElAz = numpy.repeat(camPositionsElAz,camOrientations.shape[0],axis=0)
        camOrientations  = numpy.tile(camOrientations,(numCams,1))
        numCams = len(camElAz)
        camDist = 40. # cm, TODO adjust for other objects ?

        if zRotInv == 1:
            camElAz[:,1] = 0.0  # if it is fully rotationally symmetric, the azimuth is always 0 
            
        path = "{}{}/train/".format(self.basePath,objName)
        #path = "{}{}/".format(self.basePath,objName)
        
        #raise ValueError("path {}".format(path))
        
        pat = "{}img*.png".format(path)
        print("looking for " + pat) 
        d = sorted(glob.glob(pat))
        
        assert numCams == len(d), "hey! {} != {}".format(numCams,len(d))
        
        if imgNums is not None:
            d = [d[i] for i in imgNums]  
            #camPoses = [camPoses[i,:] for i in imgNums]
            camElAz = [camElAz[i,:] for i in imgNums]
            camOrientations = [camOrientations[i,:] for i in imgNums]
        
        txt = "loading {}".format(objName)
        pbar = pb.ProgressBar(maxval=len(d),widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()
        
        for frameNum in xrange(len(d)):
            pbar.update(frameNum)
            fileName = d[frameNum]
            fn = os.path.split(fileName)[1]
            fnroot = fn[3:]
            
            imgFileName = fileName
            dptFileName = "{}dpt{}".format(path,fnroot)
            maskFileName = "{}mask{}".format(path,fnroot)

            #print("files: {}, {}".format(imgFileName,dptFileName))
            
            #dptRendered = Image.open(dptFileName)
            #dptrData = numpy.asarray(dptRendered,numpy.float32)
            dptrData = cv2.imread(dptFileName,cv2.CV_LOAD_IMAGE_UNCHANGED).astype(numpy.float32)
            if len(dptrData.shape) > 2:
                dptrData = dptrData[:,:,0] 
            
            dptrData *= 200./2**16.  # first in blender we expressed everything in dm (ie 1unit = 10cm), then we mapped 0..20 -> 0..1, then blender fills the the 16bit up. image is scaled to 0..1 -> 0..2^16
                        
                        
            maskData = cv2.imread(maskFileName,cv2.CV_LOAD_IMAGE_UNCHANGED).astype(numpy.float32)
            #print("maskData mn/mx {},{}".format(numpy.min(maskData),numpy.max(maskData)))
            #cv2.imshow("maskData",maskData/(2**16.))
            #raise ValueError("ok")
            maskData = (maskData < (2**16. - 5)).astype(numpy.uint8)
            #cv2.imshow("maskData2",maskData)
            #cv2.waitKey(0)
            #raise ValueError("ok")
            
            # DEBUG print value range
            #print("dptrData min {}, max {}".format(dptrData.min(),dptrData.max()))
             
            # DEBUG print value of center pixel       
            #h,w = dptrData.shape
            #print dptrData[h/2,w/2]
            
            # DEBUG show            
            #dptRendered = Image.fromarray(dptrData*255./8.) # to show it we scale from 0..8 to 0..255, everything beyond 8 is cut away (saturated)
            #dptRendered.show('rendered')
            
            #self.cropPatchFromDptAndImg(dptrData,dptrData,(320,240),3.)
            
            imgData = cv2.imread(imgFileName)
            imgData = imgData.astype(floatX) / 255. - 0.5 
            
            #tra = numpy.array([0,0,0])
            #pose = LinemodObjPose(relCamPos=rot[0:3],tra=tra)
            elAz = camElAz[frameNum] 
            pose = LinemodObjPose(zRotInv=zRotInv)
            pose.setAzEl(az=elAz[1], el=elAz[0], dist=camDist, lookAt=numpy.array([0.,0.,-5.]))
            
            #--------------------  
            # DEBUG - consistency check
            # check if this relCamPos calculated from the rotation matrix, set by az,el is still the same as calculated directly
            rcp = pose.relCamPos 
            camEl = -elAz[0]
            camAz = elAz[1]
            camX = numpy.sin(camAz) * numpy.cos(camEl)   # coordinates of cameras in blender ()
            camY = numpy.cos(camAz) * numpy.cos(camEl)
            camZ = numpy.sin(camEl)
            camP = numpy.array([camX,camY,camZ])*camDist
            camP += numpy.array([0.,0.,-5.]) # offset from ground 
            #print("camP {}, rcp {}".format(camP,rcp))
            if not numpy.allclose(rcp,camP):
                raise ValueError("Shit just got real: rcp={} != camP={}".format(rcp,camP)) 
            
            
            frame = Frame(imgData,dptrData,maskData,pose,objName,imgFileName)
            #dptPatch,imgPatch = self.cropPatchFromDptAndImg(dptrData,imgData,(320,240),cropSize,targetSize,fixedD=4.0)
            #seqData.append(Frame(imgPatch,dptPatch,pose,objName,imgFileName))
            patch = self.cropPatchFromFrame(frame,(320,240),cropSize,targetSize,fixedD=camDist)
            seqData.append(patch)
            
            # keep only the first few full frames for debugging
            if frameNum > 10:
                frame.img = None
                frame.dpt = None
                frame.mask = None
            
        return NamedImgSequence(objName,seqData)

            
            
    def cropPatchFromFrame(self,frame,center,cropSize,targetSize,fixedD=None):

        dptImg = frame.dpt
        img = frame.img
        mask = frame.mask
        
        #targetSize = 64.
        
        cam = BlenderCam()
        
        cx = center[0]
        cy = center[1]

        # distance at point along the z-axis
        if fixedD is None: 
            cropDepth = dptImg[cy,cx]
        else:
            cropDepth = fixedD

        # point on image plane in world space
        cx = float(cx - cam.cpx) * cam.sensor_w/cam.imgw # pixel to metric
        cy = float(cy - cam.cpy) * cam.sensor_h/cam.imgh 
                
        # center point in world coords
        wx = cx * cropDepth / cam.focus   
        wy = cy * cropDepth / cam.focus
        
        # vector cam-center (ie origin) to point in the world 
        v = numpy.array([wx,wy,cropDepth])
        
        # up vector of rectangle to crop in the world = x-axis X v = (1,0,0) X (vx,vy,vz) = (0,-vz,vy)    (or really it is down (depending on the coord system))
        u = numpy.array([0.,-cropDepth,wy])
        # right vector of rectangle to crop = u X v = (wy^2+cropDepth^2,-wx*wy,-wx*cropDepth) 
        r = numpy.array([wy*wy+cropDepth*cropDepth,-wx*wy,-wx*cropDepth])
        
        # normalize, resize
        u = u * (cropSize/2) / numpy.linalg.norm(u)
        r = r * (cropSize/2) / numpy.linalg.norm(r)
        
        # world coord of corner points
        w1 = v + u - r
        w2 = v - u - r
        w3 = v + u + r
        w4 = v - u + r
        
        # coords of corner points on image plane
        sourceRect = numpy.array([[w1[0]/w1[2],w1[1]/w1[2]],
                                  [w2[0]/w2[2],w2[1]/w2[2]],
                                  [w3[0]/w3[2],w3[1]/w3[2]],
                                  [w4[0]/w4[2],w4[1]/w4[2]]]) * cam.focus
        # convert into pixel space
        sourceRect = sourceRect * numpy.array((cam.imgw/cam.sensor_w,cam.imgh/cam.sensor_h))
        sourceRect = sourceRect + numpy.array((cam.cpx,cam.cpy))
        sourceRect = sourceRect.astype(numpy.float32)
        
        # target pts 
        targetRect = numpy.array([[0,0],[0,1],[1,0],[1,1]]).astype(numpy.float32) * targetSize 
        M = cv2.getPerspectiveTransform(sourceRect,targetRect)
        
        ts = (targetSize,targetSize)
        #dptPatch = numpy.zeros(ts,dtype=numpy.float32)
        #imgPatch = numpy.zeros(ts,dtype=numpy.float32)
        dptPatch = cv2.warpPerspective(dptImg,M,ts)
        imgPatch = cv2.warpPerspective(img,M,ts)
        mask = cv2.warpPerspective(mask,M,ts)
        
        # normalize dpt data: crop out a cube around the cropDepth
        dHigh = 20. 
        dLow = -20.
        dptPatch = dptPatch - cropDepth
        dptPatch = numpy.maximum(numpy.minimum(dptPatch,dHigh),dLow)
        dptPatch /= numpy.maximum(abs(dHigh),abs(dLow))  # -> -1..1
        
        patch = Patch(imgPatch,dptPatch,mask,frame,sourceRect)
        #return (dptPatch,imgPatch)
        return patch
    
#         print("dptPatch min {}, max {}".format(dptPatch.min(),dptPatch.max()))
#         cv2.imshow("hello",dptPatch/8.)
#         cv2.waitKey(0)
        
        #dptI = Image.fromarray(dptPatch*255./8.) # to show it we scale from 0..8 to 0..255, everything beyond 8 is cut away (saturated)
        #dptI.show('rendered')        

        # with Pillow it would be  
        #   eg: dptRendered.transform((10,10), Image.QUAD, (x0,y0,x1,y1,x2,y2,x3,y3), Image.BILINEAR)
        #   from http://effbot.org/imagingbook/image.htm
        # but we use opencv here - because no need to transform the numpy array and maybe faster ...



    def testsWithDepthMapsFromBlender(self,objName,imgNums=None):
        
        # camera data
        cam_f = 28.6 # mm, camera focus length
        cam_imgw = 640 # pixels
        cam_imgh = 480 # pixels
        cam_w = 32.  # mm,  sensor width
        cam_h = cam_w * cam_imgh / cam_imgw
        cam_cx = cam_w/2 - 0.009 # mm, camera center shift horizontal (+ or - ?)
        cam_cy = cam_h/2 + 0.003 # mm, camera center shift vertical
        
        
        path = "{}{}/train/".format(self.basePath,objName)
        #path = "{}{}/".format(self.basePath,objName)
        
        pat = "{}img*.png".format(path)
        print("looking for " + pat) 
        d = sorted(glob.glob(pat))
        if imgNums is not None:
            d = [d[i] for i in imgNums]  # rly?
        
        for fileName in d:
            fn = os.path.split(fileName)[1]
            fnroot = fn[3:]
            
            imgFileName = fileName
            dptFileName = "{}dpt{}".format(path,fnroot)

            #dptFileName = "/home/wohlhart/work/data/linemod/blender_experiments/tmp_renders/templates/dpt0314.png" 
         
            print("files: {}, {}".format(imgFileName,dptFileName))
            
            dptRendered = Image.open(dptFileName)
            dptrData = numpy.asarray(dptRendered,numpy.float32)    
            #dptrData *= 255./2**16.  # blender fills the the 16bit up. image is scaled to 0..1 -> 0..2^16 
            dptrData *= 20./2**16.  # first in blender we mapped 0..20 -> 0..1, then blender fills the the 16bit up. image is scaled to 0..1 -> 0..2^16
            print("dptrData min {}, max {}".format(dptrData.min(),dptrData.max()))
                        
            dptRendered = Image.fromarray(dptrData*255./8.) # to show it we scale from 0..8 to 0..255, everything beyond 8 is cut away (saturated)
            dptRendered.show('rendered')
            
            h,w = dptrData.shape
            print dptrData[h/2,w/2]
            
            plt.plot(dptrData[:,306])
            plt.plot(dptrData[:,200])
            plt.show()
             
            #plt.plot(dptrData[350,:])
            plt.plot(dptrData[h/2,:])
            plt.show()
            
            mx,my = numpy.meshgrid(range(dptrData.shape[1]),range(dptrData.shape[0]))
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             ax.plot_surface(mx,my,dptrData, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#             plt.show()
            
            mx = mx*cam_w/cam_imgw - cam_cx  # pixel to metric - offset
            my = my*cam_h/cam_imgh - cam_cy  # pixel to metric - offset
            #rsq = mx*mx + my*my  # distance of point on screen from image center
            #ds = numpy.sqrt(rsq + cam_f*cam_f)  # distance from camera center to point on screen
            #db = # 
            mx = mx*dptrData/cam_f
            my = my*dptrData/cam_f
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #ax.plot_surface(mx,my,dptrData, cmap=cm.coolwarm,linewidth=0, antialiased=False)  # connects over edges ...
            #ax.scatter3D(mx,my,dptrData, cmap=cm.coolwarm,linewidth=0, antialiased=False)  #SLOW
            #ax.scatter3D(mx.flatten(),my.flatten(),dptrData.flatten(), marker=".", antialiased=False)  #SLOW
            ax.plot3D(mx.flatten(),my.flatten(),'b.',zs=dptrData.flatten(),markersize=1)
            plt.show()
