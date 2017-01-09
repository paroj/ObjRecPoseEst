'''
Created on Aug 22, 2014

@author: wohlhart
'''
import numpy

class BlenderCam(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        # according to Stefan Hinterstoisser
        # fx=572.41140, px=325.26110, fy=573.57043; py=242.04899
        #
        # in blender there is only one focus -> f = (fx+fy)/2 = 573 
        # image = 640, sensor = 32
        #  -> in world coords (mm) f = 573 *32 / 640 = 573 / 20 = 28.65
        # 
        # offset: 
        #   325.2611 - (640/2) = 1.2611
        #   242.04899 - (320/2) = 2.04899
        # in world coords:
        #   1.26 / 20 = 0.063
        #   2.04899 / 20 = 0.1024495
        # in Blender the shift_x, shift_y is relative to the sensor size, also x is to the left (positive moves the camera plane to the right -> the image to the left) 
        #  shift_x = -1.26/320 = 0.0039375
        #  shift_y = 2.04/320 = 0.006403
        # however, this was not the best settings => trying around in blender delivered: -0.009, 0.003  for the best overlap with the captured images

        # camera data
        self.focus = 28.6/100 # dm, camera focus length
        self.imgw = 640 # pixels
        self.imgh = 480 # pixels
        self.sensor_w = 32.0/100  # dm,  sensor width
        self.sensor_h = self.sensor_w * self.imgh / self.imgw
        self.cx = self.sensor_w/2 - 0.009/100 # dm, camera center shift horizontal (+ or - ?)
        self.cy = self.sensor_h/2 + 0.003/100 # dm, camera center shift vertical        
        self.cpx = self.imgw/2 + 0.009*self.imgw #  cam principal point in pixels 
        self.cpy = self.imgh/2 + 0.003*self.imgw #  cam principal point in pixels
        #self.cpx = self.imgw/2  # DEBUG, no shift 
        #self.cpy = self.imgh/2 
        
    def screenToWorld(self,x,y):
        worldCoords = [] # TODO
        return worldCoords  
        
    def worldToScreen(self,pt):
        '''
        calcuate point on the screen in pixels
        for a point in space given in meters/10
        '''
        
        x = pt[0]
        y = -pt[1]  # y is down in image space
        z = -pt[2]  # looking into negative z
        
        # project onto image plane 
        x = x * self.focus / z
        y = y * self.focus / z
        
        # metric to pixels 
        x *= self.imgw/self.sensor_w
        y *= self.imgh/self.sensor_h
        
        # shift to image origin
        x += self.cpx
        y += self.cpy
        
        screenCoords = numpy.array([x,y]) 
        return screenCoords  
        
        
class LinemodCam(object):
    
    def __init__(self):
        '''
        Constructor
        '''
        
        # according to Stefan Hinterstoisser
        # fx=572.41140, px=325.26110, fy=573.57043; py=242.04899
        #
        # in blender there is only one focus -> f = (fx+fy)/2 = 573 
        # image = 640, sensor = 32
        #  -> in world coords (mm) f = 573 *32 / 640 = 573 / 20 = 28.65
        # 
        # offset: 
        #   325.2611 - (640/2) = 1.2611
        #   242.04899 - (320/2) = 2.04899
        # in world coords:
        #   1.26 / 20 = 0.063
        #   2.04899 / 20 = 0.1024495
        # in Blender the shift_x, shift_y is relative to the sensor size, also x is to the left (positive moves the camera plane to the right -> the image to the left) 
        #  shift_x = -1.26/320 = 0.0039375
        #  shift_y = 2.04/320 = 0.006403
        # however, this was not the best settings => trying around in blender delivered: -0.009, 0.003  for the best overlap with the captured images

        # camera data
        self.focus = 2.86 # cm, camera focus length
        self.imgw = 640 # pixels
        self.imgh = 480 # pixels
        self.sensor_w = 3.2  # cm,  sensor width
        self.sensor_h = self.sensor_w * self.imgh / self.imgw
        self.cx = self.sensor_w/2 - 0.009/10 # cm, camera center shift horizontal (+ or - ?)
        self.cy = self.sensor_h/2 + 0.003/10 # cm, camera center shift vertical        
        self.cpx = self.imgw/2 + 0.009*self.imgw #  cam principal point in pixels 
        self.cpy = self.imgh/2 + 0.003*self.imgw #  cam principal point in pixels
        
    def screenToWorld(self,x,y):
        worldCoords = [] # TODO
        return worldCoords  
        
    def worldToScreen(self,pt):
        '''
        calcuate point on the screen in pixels
        for a point in space given in meters/10
        '''
        
        if pt.ndim > 1:
            x = pt[:,0]
            y = pt[:,1]  # y is down in image space
            z = pt[:,2]  # looking into negative z
            nPts = pt.shape[0]
        else:
            x = pt[0]
            y = pt[1]  # y is down in image space
            z = pt[2]  # looking into negative z
        
        # project onto image plane 
        x = x * self.focus / z
        y = y * self.focus / z
        
        # metric to pixels 
        x *= self.imgw/self.sensor_w
        y *= self.imgh/self.sensor_h
        
        # shift to image origin
        x += self.cpx
        y += self.cpy
        
        if pt.ndim > 1:
            screenCoords = numpy.concatenate((x.reshape((nPts,1)),y.reshape((nPts,1))),axis=1)
        else:
            screenCoords = numpy.array([x,y]) 
        return screenCoords  
    
