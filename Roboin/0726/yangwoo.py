import cv2
import numpy as np
#import RPi.GPIO as GPIO
import picamera
import time
#from ar_markers import detect_markers
from picamera.array import PiRGBArray
import sys
import linecache

# =============================================================================
# 
# motor1A = 16
# motor1B = 18
# motor2A = 22
# motor2B = 24
# en1 = 40
# en2 = 38
# 
# echo = 7
# trigger = 11
# ###gnd=6, vcc=4
# 
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(motor1A, GPIO.OUT)
# GPIO.setup(motor1B, GPIO.OUT)
# GPIO.setup(motor2A, GPIO.OUT)
# GPIO.setup(motor2B, GPIO.OUT)
# GPIO.setup(en1,GPIO.OUT)
# GPIO.setup(en2,GPIO.OUT)
# 
# GPIO.setup(echo, GPIO.IN)
# GPIO.setup(trigger, GPIO.OUT)
# GPIO.output(trigger, False)
# 
# 
# p1A = GPIO.PWM(en1, 1000)
# p1B = GPIO.PWM(en2, 1000)
# 
# p1A.start(0)
# p1B.start(0)
# 
# =============================================================================
def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print ('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


#def measure():
# =============================================================================
#     try:
#         GPIO.output(trigger, True)
#         time.sleep(0.00001)
#         GPIO.output(trigger, False)
#         start = time.time()
#         ps = start
#         
#         while GPIO.input(echo)==0 and time.time() < ps + 0.05:
#             start = time.time()
#             
#         while GPIO.input(echo)==1 and time.time() < start + 0.1:
#             stop = time.time()
#         
#         elapsed = stop-start
#         distance = (elapsed * 34300)/2
#     except:
#         distance = -1
#     
#     return distance
#         
# =============================================================================
# # =============================================================================
# def motor(left, right):
#     left = np.clip(left, -100 , 100)
#     right = np.clip(right, -100, 100)
#     
#     if left >= 0:
#         left_f = left
#         left_b = 0
#     else:
#         left_f = 0
#         left_b = left
#     	
#     if right >= 0:
#         right_f = right
#         right_b = 0
#     else:
#         right_f = 0
#         right_b = right
#         
#     if left_f > 0 and right_f > 0:
#         p1A.ChangeDutyCycle(left_f)
#         p1B.ChangeDutyCycle(right_f)
#         GPIO.output(motor1A,True)
#         GPIO.output(motor1B,False)
#         GPIO.output(motor2A,False)
#         GPIO.output(motor2B,True)
#         
#     elif left_f == 0 and left_b == 0 and right_f == 0 and right_b == 0:
#         p1A.ChangeDutyCycle(left_f)
#         p1B.ChangeDutyCycle(right_f)
#         GPIO.output(motor1A,True)
#         GPIO.output(motor1B,False)
#         GPIO.output(motor2A,False)
#         GPIO.output(motor2B,True)       
#         
#     elif left_f > 0 and right_b < 0:
#         p1A.ChangeDutyCycle(left_f)
#         p1B.ChangeDutyCycle(-right_b)
#         GPIO.output(motor1A,True)
#         GPIO.output(motor1B,False)
#         GPIO.output(motor2A,True)
#         GPIO.output(motor2B,False)    
#         
#     elif left_b < 0 and right_b <0 :
#         p1A.ChangeDutyCycle(-left_b)
#         p1B.ChangeDutyCycle(-right_b)
#         GPIO.output(motor1A,False)
#         GPIO.output(motor1B,True)
#         GPIO.output(motor2A,True)
#         GPIO.output(motor2B,False)
# 
# =============================================================================
# # =============================================================================
# def detectTUMB(cascade_classifier, gray_image, image):
# 
#     mode="No object"
#     cascade_obj = cascade_classifier.detectMultiScale(
#         gray_image,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(10,10),
#         maxSize=(60,60)
#         #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
#     )
#     for (x_pos, y_pos, width, height) in cascade_obj:
#         # draw a rectangle around the objects
#         if(width>=10):
#             cv2.rectangle(image, (x_pos, y_pos), (x_pos+width, y_pos+height), (255, 255, 255), 2)
#             cv2.putText(image, 'STOP', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             mode="STOP"
#     return mode
# 
# cas_time= time.time()
# start_time = time.time()
# cup_cascade=cv2.CascadeClassifier('cascade.xml')        
# map1 = np.load('map1.npy')
# map2 = np.load('map2.npy')
# 
# =============================================================================
# # =============================================================================
# def undistort(img):
#     h,w = img.shape[:2]
#     undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#     return undistorted_img
# =============================================================================

def reg_of_int(image):
    #region of interest
    region=np.array([[[0,120],[105,90],[215,90],[320,120],[320,240],[0,240]]],dtype=np.int32)
    #un masked region
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,region,(255,255,255))#mask
    
    new_image=cv2.bitwise_and(image,mask)#merge image
    return new_image    
# =============================================================================
# 
# def ar_markers(markers_id):
#     ans = 'forward'
#     if markers_id == 2537:
#         ans = 'stop'
#     elif markers_id == 114:
#         ans = 'lleft'
#     elif markers_id == 1156:
#         ans = 'rright'
#     elif markers_id == 923:
#         ans = 'lleft'    
#     return ans
# 
# =============================================================================
        
        
def select_white(image, white):
    lower = np.uint8([white,white,white])
    upper = np.uint8([255,255,255])
    white_mask = cv2.inRange(image, lower, upper)
    return white_mask

def select_color(image, lower, upper):
    image_HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return cv2.inRange(image_HSV,lower,upper)

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
camera.resolution = (320, 240)
camera.vflip = True
camera.hflip = True
camera.framerate = 15

rawCapture = PiRGBArray(camera, size=(320,240))


try:
    mode = "No object"
    while True:
        
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            start_time = time.time()
            image = frame.array
        
            real = image
# =============================================================================
#             if start_time - cas_time > 1:               
#                 gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
#                 mode =detectTUMB(cup_cascade, gray, real)
# =============================================================================
            obstacle = mode

    
# =============================================================================
#             markers = detect_markers(undistort(real))
#             detect=0
#             for marker in markers:
#                 detect = marker.id
#                 marker.highlite_marker(real)
#         
#             ans = ar_markers(detect)
# =============================================================================
    
            masked_image = select_white(real,150)
            mask_add_image = reg_of_int(masked_image)
    
            command = set_path(mask_add_image, 0.1)
            direction = command[0]
        

            image_show = mask_add_image
            y1, x1 = mask_add_image.shape
            x1 = int(x1/2)
            x2 = int(-command[2]*command[1] + x1)
            y2 = y1-command[2]
            cv2.line(image_show,(x1,y1),(x2,y2),(255,255,255),2)

                    
            image__show = mask_add_image
            font = cv2.FONT_HERSHEY_SIMPLEX
            text0 = command[0]
            text1 = 'ratio: ' + str(command[1])
            text2 = 'distance: ' + str(command[2])
            cv2.putText(image__show, text0, (200,20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(image__show, text1, (10,20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(image__show, text2, (200,40), font, 0.5, (255, 255, 255), 1)    
        
            rawCapture.truncate(0)

            #distance = measure()
            
    
            if obstacle == 'No object':

                #if ans == 'forward':                                
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
# =============================================================================
#                 elif ans == 'rright':
#                     for i in range(1,6201):
#                         motor(10,53)
#                         print(i)
#                     for i in range(1,2501):
#                         motor(53,53)
#                         print(i)
#                             
#                 elif ans == 'lleft':
#                     for i in range(1,5501):
#                         motor(53,10)
#                         print(i)
# 
#                 elif ans == 'stop':
#                     for i in range(1,300001):
#                         motor(0,0)
#                         print(i)
# =============================================================================
            #else:
# =============================================================================
#                 motor(0,0)
#             elif obstacle == "STOP":
#                 motor(0,0)
#                 time.sleep(5)
#                 for i in range(1,5001):
#                     motor(53,53)
#                     print(i)
# =============================================================================

                    
            #cv2.imshow('mask_image',mask_add_image)
            cv2.imshow('mask_image',image)
            rawCapture.truncate(0)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
    cv2.destroyAllWindows()
except:
    pass