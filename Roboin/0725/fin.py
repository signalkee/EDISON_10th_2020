import cv2
import numpy as np
#import RPi.GPIO as GPIO
import picamera
import time
from ar_markers import detect_markers
from picamera.array import PiRGBArray
import sys
import linecache
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def thres_roi_mask(image, white):
    lower = np.uint8([white,white,white])
    upper = np.uint8([255,255,255])
    white_mask = cv2.inRange(image, lower, upper)
    height, width = white_mask.shape

    mask = np.zeros_like(white_mask)
    region_of_interest_vertices = np.array([[1,height-1],[1,180], [int(width/3), 140],[int(width*2/3), 140], [width-1, 180], [width-1,height-1]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], (255,255,255))
    thresholded = cv2.bitwise_and(white_mask, mask)

    return thresholded 

def first_nonzero(arr, axis, invalid_val=-1):
    arr = np.flipud(arr)
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def set_path(image, forward_criteria):
    height, width = image.shape
    height = height-1
    width = width-1
    center=int(width/2)
    left=0
    right= width
    
    center = int((left+right)/2)
    
    try:
        if image[height][:center].min(axis=0) == 255:
            left = 0
        else:
            left = image[height][:center].argmin(axis=0)
        if image[height][center:].max(axis=0) ==0:
            right = width
        else:
            right = center+image[height][center:].argmax(axis=0)
        
        center = int((left+right)/2)
        
        forward =int(first_nonzero(image[:,center],0,height))-1
        
        left_line = first_nonzero(image[height-forward:height,center:],1, width-center)
        right_line = first_nonzero(np.fliplr(image[height-forward:height,:center]),1, center)
        
        center_y = (np.ones(forward)*2*center-left_line+right_line)/2-center
        center_x = np.vstack((np.arange(forward), np.zeros(forward)))
        m, c = np.linalg.lstsq(center_x.T,center_y, rcond=-1)[0] 

    except:
        result = 'backward'
        m = 0
    
    return result, round(m,4), forward

def purePursuit(lookAheadPtX, lookAheadPtY):
    ratio_s2w = 1 # steer & wheel angle ratio
    L = 0.2 # front wheel base -> rear wheel base
    Lfw = 0.3 # change smaller-exact follow (much harder)
    
    eta = math.atan2(lookAheadPtY, lookAheadPtX)
    Lw = math.sqrt(lookAheadPtX**2 + lookAheadPtY**2)
    
    steerAngle = ratio_s2w*math.atan((L * math.sin(eta)) / (Lw*1.5 / 2 + Lfw*math.cos(eta)))
    
    max_steering_angle = math.radians(90.0)
    if abs(steerAngle) > max_steering_angle:
        steerAngle = np.sign(steerAngle) * max_steering_angle
    
    return steerAngle

def ang2vel (angle):
    fullspeed = 100
    constlow = 60 #changable
    consthigh = 20 #changable
    if angle < 0:
        left = fullspeed - abs(angle)*constlow
        right = fullspeed # - abs(angle)*consthigh
    elif angle == 0:
        left = fullspeed
        right = fullspeed
    else:
        left = fullspeed # -abs(angle)*consthigh
        right = fullspeed - abs(angle)*constlow
    return left, right


camera = picamera.PiCamera()
camera.rotation = 90
camera.resolution = (320, 240)
camera.vflip = True
camera.hflip = True
camera.framerate = 15

rawCapture = PiRGBArray(camera, size=(320,240))

try:
    while True:
        
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            
            image = frame.array
            real_img = image

            thres_roi_mask_img = thres_roi_mask(real_img,190)

    
            command = set_path(thres_roi_mask_img, 0.1)
            

            # showing Line of direction
            y1, x1 = thres_roi_mask_img.shape
            x1 = int(x1/2)
            x2 = int(-command[2]*command[1] + x1)
            y2 = int(y1-command[2])
            dx = x2-x1
            dy = y1-y2
            print('x2: {}'.format(x2))
            print('y2: {}'.format(y2))
            cv2.circle(real_img,(x2,y2),5,(0,0,255),3)
            #cv2.circle(thres_roi_mask_img,(x2,y2),5,(255,255,255),3)
            

            steer_angle = purePursuit(dx, dy)
            left, right = ang2vel(steer_angle)
            print("left: {} // right: {} ".format(left, right))

            font = cv2.FONT_HERSHEY_SIMPLEX
            text1 = 'x dist: ' + str(dx)
            text2 = 'y dist: ' + str(dy)
            cv2.putText(real_img, text1, (10,20), font, 0.5, (0, 0, 255), 1)
            cv2.putText(real_img, text2, (10,40), font, 0.5, (0, 0, 255), 1) 
            #cv2.putText(thres_roi_mask_img, text1, (10,20), font, 0.5, (255, 255, 255), 1)
            #cv2.putText(thres_roi_mask_img, text2, (10,40), font, 0.5, (255, 255, 255), 1) 

            rawCapture.truncate(0)
      
    

            # if direction == 'forward':
            #     #motor(100,70)
            #     print('forward')
            # elif direction == 'left':
            #     #motor(54,30)
            #     print('left')
            # elif direction == 'right':
            #     #motor(45,54)
            #     print('right')
            # elif direction == 'backward':
            #     #motor(-40,-40)
            #     print('backward')
            # elif direction == 'lleft':
            #     #motor(45,30)
            #     print('lleft')
            # elif direction == 'rright':
            #     #motor(30,45)
            #     print('rright')
            # elif direction == 'llleft':
            #     #motor(53,20)
            #     print('llleft')
            # elif direction == 'rrright':
            #     #motor(20,53)
            #     print('rrright')
            # elif direction == 'uturn':
            #     #motor(40,-40)
            #     print('uturn')

            cv2.imshow('show',real_img)
            #cv2.imshow('show',thres_roi_mask_img)
            rawCapture.truncate(0)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
    cv2.destroyAllWindows()
except:
    pass