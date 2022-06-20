import cv2
import numpy as np
#import RPi.GPIO as GPIO
import picamera
import time
#from ar_markers import detect_markers
from picamera.array import PiRGBArray
import sys
import linecache



def select_white(image, white):
    lower = np.uint8([white,white,white])
    upper = np.uint8([255,255,255])
    white_mask = cv2.inRange(image, lower, upper)
    return white_mask

def reg_of_int(image):
    #region of interest
    region=np.array([[[0,120],[105,90],[215,90],[320,120],[320,240],[0,240]]],dtype=np.int32)
    #un masked region
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,region,(255,255,255))#mask
    
    new_image=cv2.bitwise_and(image,mask)#merge image
    return new_image    

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
        
        if abs(m) < forward_criteria and forward > 45:
            result = 'forward'
        elif forward <= 45 and forward >=20 and abs(m) < forward_criteria+0.03 :
            result = 'uturn'
        elif forward_criteria <= abs(m) and abs(m) < 0.13 and m > 0:
            result = 'left'
        elif abs(m) > 0.13 and abs(m) < 0.24 and m > 0:
            result = 'lleft'
        elif abs(m) >= 0.24 and m > 0:
            result = 'llleft'
        elif forward_criteria <= abs(m) and abs(m) < 0.30 and m < 0:
            result = 'right'
        elif abs(m) > 0.30 and abs(m) < 0.46 and m < 0:
            result = 'rright'
        elif abs(m) >= 0.46 and m < 0:
            result = 'rrright'
        else:
            result = 'backward'
    except:
        result = 'backward'
        m = 0
    
    return result, round(m,4), forward








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
            real = image    
            masked_image = select_white(real,150)
            mask_add_image = reg_of_int(masked_image)
    
            command = set_path(mask_add_image, 0.1)
            direction = command[0]

            # showing Line of direction
            image_show = mask_add_image
            y1, x1 = mask_add_image.shape
            x1 = int(x1/2)
            x2 = int(-command[2]*command[1] + x1)
            y2 = y1-command[2]
            cv2.line(image_show,(x1,y1),(x2,y2),(255,255,255),5)

            image__show = mask_add_image
            font = cv2.FONT_HERSHEY_SIMPLEX
            text0 = command[0]
            text1 = 'ratio: ' + str(command[1])
            text2 = 'distance: ' + str(command[2])
            cv2.putText(image__show, text0, (200,20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(image__show, text1, (10,20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(image__show, text2, (200,40), font, 0.5, (255, 255, 255), 1)    
            rawCapture.truncate(0)
      
    

            if direction == 'forward':
                #motor(100,70)
                print('forward')
            elif direction == 'left':
                #motor(54,30)
                print('left')
            elif direction == 'right':
                #motor(45,54)
                print('right')
            elif direction == 'backward':
                #motor(-40,-40)
                print('backward')
            elif direction == 'lleft':
                #motor(45,30)
                print('lleft')
            elif direction == 'rright':
                #motor(30,45)
                print('rright')
            elif direction == 'llleft':
                #motor(53,20)
                print('llleft')
            elif direction == 'rrright':
                #motor(20,53)
                print('rrright')
            elif direction == 'uturn':
                #motor(40,-40)
                print('uturn')




            cv2.imshow('mask_image',mask_add_image)
            rawCapture.truncate(0)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
    cv2.destroyAllWindows()
except:
    pass