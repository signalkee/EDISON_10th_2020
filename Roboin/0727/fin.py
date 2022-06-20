import cv2
import numpy as np
import RPi.GPIO as GPIO
import picamera
import time
from ar_markers import detect_markers
from picamera.array import PiRGBArray
import sys
import linecache
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def detect_marker(image):
    markers = detect_markers(image)
    for marker in markers:
        if marker.id == 114: #left - change
            print("ARmarker id:  %d  ==> To the left" % marker.id)
            #time.sleep(0.7)
            marker.highlite_marker(image)
            motor(-50,50)
            #time.sleep(1)
        elif marker.id == 1156: #right - change
            print("ARmarker id:  %d  ==> To the right" % marker.id)
            #time.sleep(0.7)
            marker.highlite_marker(image)
            motor(50,-50)
            #time.sleep(1)

def ROI_mask(img):
    height, width = img.shape[0], img.shape[1]
    print(height)
    print(width)
    fill_color = [0, 0, 0] 
    mask_value = 255
    contours = [ np.array([[1,height-1],[1,180], [int(width/3), 140],[int(width*2/3), 140], [width-1, 180], [width-1,height-1]])]
    stencil = np.zeros(img.shape[:-1]).astype(np.uint8)
    print('777777')
    cv2.fillPoly(stencil, contours, mask_value)
    print(img)
    print('stemci;')
    print(stencil)
    sel = stencil != mask_value
    img[sel] = fill_color
    print('sel')
    print(sel)
    print('image_sel')
    print(img[sel])
    
    
    print('567567567')
    
    
    return real
    

def thres_roi_mask(image, white):
    lower = np.uint8([white,white,white])
    upper = np.uint8([255,255,255])
    white_mask = cv2.inRange(image, lower, upper)
    height, width = white_mask.shape

    mask = np.zeros_like(white_mask)
    # region_of_interest_vertices = np.array([[1,height-1],[1,180], [int(width/3), int(height/3)],[int(width*2/3), int(height/3)], [width-1, 180], [width-1,height-1]], dtype=np.int32)
    region_of_interest_vertices = np.array([[1,height-1],[1,int(height/2)], [width-1, int(height/2)], [width-1,height-1]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], (255,255,255))
    thresholded = cv2.bitwise_and(white_mask, mask)

    return thresholded 

def first_nonzero(arr, axis, invalid_val=-1):
    arr = np.flipud(arr)
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def set_path(image):
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
        m = 0
    
    return round(m,4), forward

def purePursuit(lookAheadPtX, lookAheadPtY):
    ratio_s2w = 1 # steer & wheel angle ratio
    L = 0.2# front wheel base -> rear wheel base
    Lfw = 0.03 # change smaller-exact follow (much harder)
    lookAheadPtX, lookAheadPtY = lookAheadPtX/1000, lookAheadPtY/1000
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
        left = fullspeed - int(abs(angle)*constlow)
        right = fullspeed # - abs(angle)*consthigh
    elif angle == 0:
        left = fullspeed
        right = fullspeed
    else:
        left = fullspeed # -abs(angle)*consthigh
        right = fullspeed - int(abs(angle)*constlow)
    return left, right

def xy2vel (dx, dy):
    lth = math.sqrt(dx**2+dy**2)
    fullspeed = lth*50/140
    if fullspeed <= 35:
        fullspeed = 35
    if dy <= 5:
        left = -fullspeed
        right = -fullspeed
    else:
        if dx == 0:
            left, right = fullspeed, fullspeed
        elif dx < 0:
            left = fullspeed-(abs(dx)*1/5)
            right = fullspeed
        else:
            left = fullspeed
            right = fullspeed-(abs(dx)*1/5)
            
    
    return left, right
        

def motor(left, right):
    left = np.clip(left, -100 , 100)
    right = np.clip(right, -100, 100)
    
    if left >= 0:
        left_f = left
        left_b = 0
    else:
        left_f = 0
        left_b = left
        
    if right >= 0:
        right_f = right
        right_b = 0
    else:
        right_f = 0
        right_b = right
        
    if left_f > 0 and right_f > 0:
        p1A.ChangeDutyCycle(left_f)
        p1B.ChangeDutyCycle(right_f)
        GPIO.output(motor1A,True)
        GPIO.output(motor1B,False)
        GPIO.output(motor2A,False)
        GPIO.output(motor2B,True)
        
    elif left_f == 0 and left_b == 0 and right_f == 0 and right_b == 0:
        p1A.ChangeDutyCycle(left_f)
        p1B.ChangeDutyCycle(right_f)
        GPIO.output(motor1A,True)
        GPIO.output(motor1B,False)
        GPIO.output(motor2A,False)
        GPIO.output(motor2B,True)       
        
    elif left_f > 0 and right_b < 0:
        p1A.ChangeDutyCycle(left_f)
        p1B.ChangeDutyCycle(-right_b)
        GPIO.output(motor1A,True)
        GPIO.output(motor1B,False)
        GPIO.output(motor2A,True)
        GPIO.output(motor2B,False)    
        
    elif left_b < 0 and right_b <0 :
        p1A.ChangeDutyCycle(-left_b)
        p1B.ChangeDutyCycle(-right_b)
        GPIO.output(motor1A,False)
        GPIO.output(motor1B,True)
        GPIO.output(motor2A,True)
        GPIO.output(motor2B,False)

motor1A = 8
motor1B = 10
motor2A = 5
motor2B = 3
en1 = 12
en2 = 7

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings (False)
GPIO.setup(motor1A, GPIO.OUT)
GPIO.setup(motor1B, GPIO.OUT)
GPIO.setup(motor2A, GPIO.OUT)
GPIO.setup(motor2B, GPIO.OUT)
GPIO.setup(en1,GPIO.OUT)
GPIO.setup(en2,GPIO.OUT)

p1A = GPIO.PWM(en1, 1000)
p1B = GPIO.PWM(en2, 1000)

p1A.start(0)
p1B.start(0)

camera = picamera.PiCamera()
camera.resolution = (320, 224)
camera.vflip = True
camera.hflip = True
camera.framerate = 15

rawCapture = PiRGBArray(camera, size=(320,224))

try:
    while True:
        
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            #print('11111')
            image = frame.array
            real_img = image
            #print('dddd')

            thres_roi_mask_img = thres_roi_mask(real_img,120)
            #print('11111')
            
            #real_img = ROI_mask(real_img)
            
    
            command = set_path(thres_roi_mask_img)
            #print('2222')

            # showing Line of direction
            y1, x1 = thres_roi_mask_img.shape
            x1 = int(x1/2)
            x2 = int(-command[1]*command[0] + x1)
            y2 = int(y1-command[1])
            dx = x2-x1
            dy = y1-y2
            # angle = -(atan2(dy/dx)-(math.pi/2))
            # print(angle)
            # print('x2: {}'.format(dx))
            # print('y2: {}'.format(dy))
            cv2.circle(real_img,(x2,y2),5,(0,0,255),3)
            #cv2.circle(thres_roi_mask_img,(x2,y2),5,(255,255,255),3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text1 = 'x dist: ' + str(dx)
            text2 = 'y dist: ' + str(dy)
            cv2.putText(real_img, text1, (10,20), font, 0.5, (0, 0, 255), 1)
            cv2.putText(real_img, text2, (10,40), font, 0.5, (0, 0, 255), 1) 
            #cv2.putText(thres_roi_mask_img, text1, (10,20), font, 0.5, (255, 255, 255), 1)
            #cv2.putText(thres_roi_mask_img, text2, (10,40), font, 0.5, (255, 255, 255), 1) 

            rawCapture.truncate(0)
            #print('222222')
            
            detect_marker(real_img)
            #print('33333333')
            
            # if dx is not None and dy is not None: 
            #     steer_angle = purePursuit(dx, dy)
            #     print('steer_angle: {}'.format(steer_angle))
            #     left, right = ang2vel(steer_angle)
            #     print("left: {} // right: {} ".format(left, right))
            #     motor(left,right)
            # else:
            #     print('backward')
            #     motor(-50,-50)
            # if angle is not None:
            #     left,right = ang2vel(steer_angle)
            if dx is not None and dy is not None: 
                #dx = int(dx)
                #dy = int(dy)
                left , right = xy2vel(dx,dy)
                print("left: {} // right: {} ".format(left, right))
                if left < 25 and right < 25:
                    print('vel under zero')
                    motor(-30,-30)
                    time.sleep(0.5)
                if abs(left-right) > 10:
                    if left > right:
                        left = 40
                        right = 0
                    else:
                        left = 0
                        right = 40
                    
                motor(left,right)
            else:
                print('backward')
                motor(-30,-30)

            #cv2.imshow('show',thres_roi_mask_img)
            cv2.imshow('show',real_img)
            rawCapture.truncate(0)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
    cv2.destroyAllWindows()
except:
    pass