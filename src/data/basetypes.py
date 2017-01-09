'''
Created on Jul 1, 2014

@author: wohlhart
'''
from collections import namedtuple
import numpy
from util.transformations import euler_matrix

#Frame = namedtuple('Frame',['img','dpt','pose','className','filename'])
NamedImgSequence = namedtuple('NamedImgSequence',['name','data'])
PairIdx = namedtuple('PairIdx',['idx0','idx1','labels'])

class Frame(object):
    
    def __init__(self,img=None,dpt=None,mask=None,pose=None,className=None,filename=None):
        self.img = img
        self.dpt = dpt
        self.mask = mask
        self.pose = pose
        self.className = className
        self.filename = filename
        
        
class LinemodTestFrame(Frame):
        
    def __init__(self,img=None,dpt=None,mask=None,pose=None,className=None,filename=None):
        super(LinemodTestFrame,self).__init__(img=img,dpt=dpt,mask=mask,pose=pose,className=className,filename=filename)
        
        
    
class Patch(object):
    '''
    A patch cropped from a Frame
    '''
    def __init__(self,img=None,dpt=None,mask=None,frame=None,cropArea=None):
        self.img = img
        self.dpt = dpt
        self.mask = mask
        self.frame = frame
        self.cropArea = cropArea   

    #def cropFromFrame(frame,cropArea):   # TODO create the img and dpt, or just keep as references? 
    
class PairStacks(object):
    '''
    Struct to hold two stacks of image data and a label indicating if the individual pairs are equal/similar or unequal/dissimilar
    '''
    def __init__(self,x0,x1,y):
        self.x0 = x0
        self.x1 = x1
        self.y = y
        
    def normalize(self,mean,std):
        self.x0 -= mean
        self.x0 /= std
        self.x1 -= mean
        self.x1 /= std
        


class LinemodObjPose(object):
    
    def __init__(self,rot=None,tra=None,zRotInv=None):
        
        self.zRotInv = zRotInv
        self.rot = rot
        self.tra = tra
        
        if rot is None:
            self.mat_rot = euler_matrix(-numpy.pi/2,0,0,axes='sxyz') # numpy.eye(4) 
        else:
            if (rot.ndim == 2):
                if (rot.shape[0] == 4) and (rot.shape[1] == 4): 
                    self.mat_rot = numpy.copy(rot)
                elif (rot.shape[0] == 3) and (rot.shape[1] == 3):
                    self.mat_rot = numpy.eye(4)
                    self.mat_rot[0:3,0:3] = rot
                else:
                    raise ValueError("rot has wrong shape {}".format(rot.shape))
            else:
                raise ValueError("rot has wrong shape {}".format(rot.shape))
       
        if tra is None:
            self.mat_trans = numpy.eye(4) 
            self.mat_trans[0:3,3] = numpy.array([0,0,1])  # set something here, otherwise we cannot compute inverse
        else:
            tra = numpy.copy(tra).squeeze()
            if tra.ndim == 1: # tra is a vector
                self.mat_trans = numpy.eye(4)
                self.mat_trans[0:3,3] = tra  
            elif (tra.ndim == 2) and (tra.shape[0] == 4) and (tra.shape[1] == 4):
                self.mat_trans = tra  
            else:
                raise ValueError("tra has wrong shape {}".format(tra.shape))            
            
        self.mat = numpy.dot(self.mat_trans,self.mat_rot)

        self.updateRelCamPos()
                
    def setAzEl(self,az,el,dist=1.0,lookAt=numpy.array([0.,0.,-5.])):
        # to move into el=0,az=0 position we first have to rotate 45deg around x
        camRot0 = euler_matrix(-numpy.pi/2,0,0,axes='sxyz')
        #camRot0 = numpy.eye(4)
        #print(numpy.dot(camRot0,numpy.array([[0,-1,0,1.]]).T))
        #raise ValueError("stop here")

        #camRot = euler_matrix(el,0,-az,axes='sxyz')
        camRot = euler_matrix(az,0,el,axes='szyx')
        self.mat_rot = numpy.dot(camRot0,camRot)
        self.mat_trans = numpy.eye(4)
        self.mat_trans[0:3,3] = [0,0,dist] 
        self.mat = numpy.dot(self.mat_trans,self.mat_rot)
        
        mat_trans0 = numpy.eye(4)
        mat_trans0[0:3,3] = -lookAt
        self.mat = numpy.dot(self.mat,mat_trans0)
        
        self.updateRelCamPos()
        
        
    def updateRelCamPos(self):
        # set relcampos (x,y,z) from the matrix
        imat = numpy.linalg.inv(self.mat) 
        #print("mat {}".format(self.mat))
        #print("imat {}".format(imat))
        #print("mat*imat {}".format(numpy.dot(self.mat,imat)))
        self._relCamPos = imat[0:3,3]   # equals dot(imat,[0,0,0,1])  which is where does the camera center (0,0,0) end up in object space  

        # z-axis rotation invariant pose:
        if self.zRotInv == 1:
            y = numpy.sqrt(self._relCamPos[0]**2 + self._relCamPos[1]**2)
            eye = numpy.array([0.,y,self._relCamPos[2]])
            target = numpy.array([0.,0.,0.])   # should the target here also be 0,0,-5   -> or the point we really look at, defined by mat?
            if not numpy.allclose(eye, target):
                if numpy.allclose(eye[:2],target[:2]):
                    up = numpy.array([0.,-1.,0.])   # eye is directly above target. so up can't be the z-axis 
                else:
                    up = numpy.array([0.,0.,-1.])
                m2 = self.lookAtMat(eye, target, up)
                im2 = numpy.linalg.inv(m2)
                self._relCamPosZRotInv = im2[0:3,3]
#                 print("eye {}".format(eye))
#                 print("target {}".format(target))
#                 print("up {}".format(up))
#                 print("m2 {}".format(m2))
#                 print("im2 {}".format(im2))
#                 print("self.mat {}".format(self.mat))                
            else:
                # something is wrong. the eye is at the origin. lets just bail (run to the hills ...)
                self._relCamPosZRotInv = self._relCamPos
        else:
            self._relCamPosZRotInv = self._relCamPos
        
        if numpy.any(numpy.isnan(self._relCamPosZRotInv)):
            raise ValueError("NaN in _relCamPos: {}, (zRotInv: {})".format(self._relCamPosZRotInv,self.zRotInv))
        
    def getMatrix(self):
        # matrix mat transforms a point in the object space into the camera space
        # assuming a linemod camera, ie one that sits at (0,0,0), looks into positive z, with x right and y down  
        return self.mat

    @property        
    def relCamPos(self):
        # where is 0,0,0 of the camera
        return self._relCamPos
        
    @relCamPos.setter
    def relCamPos(self,value):
        # input defines a camera that sits at x,y,z and has the up vector aligned with the z-axis
        
        #print("set to: x,y,z {}".format(value))
        
        eye = value
        target = numpy.array([0.,0.,0.])   # should the target here also be 
        up = numpy.array([0.,0.,-1.])
        self.mat = self.lookAtMat(eye, target, up)
        
        self.updateRelCamPos()
        
        #print("was set: x,y,z {}".format(self._relCamPos))
        #print("   mat {}".format(self.mat))
    
    @property
    def relCamPosZRotInv(self):
        return self._relCamPosZRotInv
    
    def lookAtMat(self,eye,target,up):
        # http://3dgep.com/understanding-the-view-matrix/, lookAt

        zaxis = eye - target                  # The "forward" vector.
        zaxis = zaxis / numpy.linalg.norm(zaxis)
        xaxis = numpy.cross(up, zaxis)        # The "right" vector.
        xaxis = xaxis / numpy.linalg.norm(xaxis)
        yaxis = numpy.cross(zaxis, xaxis)     # The "up" vector.
 
        # Create a 4x4 orientation matrix from the right, up, and forward vectors
        # This is transposed which is equivalent to performing an inverse
        # if the matrix is orthonormalized (in this case, it is).
        orientation = numpy.array([[xaxis[0], xaxis[1], xaxis[2], 0. ],
                                   [yaxis[0], yaxis[1], yaxis[2], 0. ],
                                   [zaxis[0], zaxis[1], zaxis[2], 0. ],
                                   [ 0., 0., 0., 1. ]])
     
        # Create a 4x4 translation matrix.
        # The eye position is negated which is equivalent
        # to the inverse of the translation matrix.
        # T(v)^-1 == T(-v)
        translation = numpy.eye(4)
        translation[0:3,3] = -eye  
 
        return numpy.dot( orientation, translation )
        
                    