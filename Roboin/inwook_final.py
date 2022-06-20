import RPi.GPIO as GPIO
import cv2
import numpy as np
import time
from ar_markers import detect_markers

from picamera import PiCamera
from picamera.array import PiRGBArray

def first_nonzero(arr, axis, invalid_val=-1):
    arr = np.flipud(arr)
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

# stopxml = cv2.CascadeClassifier('./xml/24.xml')
# uturnxml = cv2.CascadeClassifier('./xml/uturn2.xml')

def detect_marker():
    markers = detect_markers(image)
    for marker in markers:
        # for pos[[x1,y1]] in marker.contours:
        #     print(x1, y1)
            
            
        # if marker.id == 2537: #stop
        #     time.sleep(0.5)
        #     stop5()
        #     time.sleep(15)
            
                
        if marker.id == 114: #left
            print(marker.id)
            time.sleep(0.7)
            GPIO.output(motor1A, False)
            GPIO.output(motor1B, True)
            pwm1.ChangeDutyCycle(0)
            GPIO.output(motor2A, True)
            GPIO.output(motor2B, False)
            pwm2.ChangeDutyCycle(100)
            time.sleep(1)
            
        # elif marker.id == 923: #left
        #     print(marker.id)
        #     time.sleep(0.7)
        #     GPIO.output(motor1A, False)
        #     GPIO.output(motor1B, True)
        #     pwm1.ChangeDutyCycle(30)
        #     GPIO.output(motor2A, True)
        #     GPIO.output(motor2B, False)                
        #     pwm2.ChangeDutyCycle(50)
        #     time.sleep(1)
                    
        # elif marker.id == 1156: #right
        #     print(marker.id)
        #     time.sleep(0.7)
        #     GPIO.output(motor1A, True)
        #     GPIO.output(motor1B, False)
        #     pwm1.ChangeDutyCycle(50)
        #     GPIO.output(motor2A, False)
        #     GPIO.output(motor2B, True)
        #     pwm2.ChangeDutyCycle(0)
        #     time.sleep(0.8)


def detect_marker_stop():
    markers = detect_markers(image)
    for marker in markers:
       
        if marker.id == 114: #left
            print(marker.id)
            
            GPIO.output(motor1A, False)
            GPIO.output(motor1B, True)
            pwm1.ChangeDutyCycle(0)
            GPIO.output(motor2A, True)
            GPIO.output(motor2B, False)
            pwm2.ChangeDutyCycle(100)
            time.sleep(0.3)

def select_white(image, white):
   lower = np.array([white,white,white])
   upper = np.array([255,255,255])
   white_mask = cv2.inRange(image, lower, upper)
   return white_mask

def set_path1(image, upper_limit, fixed_center = 'False'):
    height, width = image.shape
    height = height-1
    width = width-1
    center=int(width/2)
    left=0
    right=width
    white_distance = np.zeros(width)
    delta_w = 10 
    delta_h = 3  
    

    if not fixed_center: 
        for i in range(center):
            if image[height,center-i] > 200:
                left = center-i
                break            
        for i in range(center):
            if image[height,center+i] > 200:
                right = center+i
                break    
        center = int((left+right)/2)      


    for i in range(int((center-left)/delta_w)+1):
        for j in range(int(upper_limit/delta_h)):
            if image[height-j*delta_h, center-i*delta_w]>200 or j==int(upper_limit/delta_h)-1: 
                white_distance[center-i*delta_w] = j*delta_h
                break        
    for i in range(int((right-center-1)/delta_w)+1):
        for j in range(int(upper_limit/delta_h)):
            if image[height-j*delta_h, center+1+i*delta_w] > 200 or j==int(upper_limit/delta_h)-1:
                white_distance[center+1+i*delta_w] = j*delta_h
                break
    
    left_sum = np.sum(white_distance[left:center])
    right_sum = np.sum(white_distance[center:right])
    forward_sum = np.sum(white_distance[center-10:center+10])
    
    if left_sum > right_sum + 500: 
            result = 'left'
    elif left_sum < right_sum - 500:
        result = 'right'
    elif forward_sum > 300: 
        result = 'forward'
    elif forward_sum > 100: 
        
        if left_sum > right_sum + 100: 
            result = 'left'
        elif left_sum < right_sum - 100:
            result = 'right'
        else:
            result = 'forward'
    # elif left_sum < 100 and right_sum in range(100,200): 
    #     result = 'uturn'
    else: 
        result = 'backward'

    
    return result, forward_sum, left_sum, right_sum


# def measure():
#     GPIO.output(GPIO_TRIGGER, True)
#     time.sleep(0.00001)
#     GPIO.output(GPIO_TRIGGER, False)
#     start = time.time()
#     timeOut = start

#     while GPIO.input(GPIO_ECHO)==0:
#         start = time.time()
#         if time.time()-timeOut > 0.05:
#             return -1

#     while GPIO.input(GPIO_ECHO)==1:
#         if time.time()-start > 0.05:
#             return -1
#         stop = time.time()
    
#     try:
#         elapsed = stop-start
#         distance = (elapsed * 34300)/2
#     except Exception as e:
#         print(e)
#         return -1

#     return distance

def forward(left, right):  
    GPIO.output(motor1A, True)
    GPIO.output(motor1B, False)
    pwm1.ChangeDutyCycle(70) #left
    GPIO.output(motor2A, True)
    GPIO.output(motor2B, False)
    pwm2.ChangeDutyCycle(78) #right

def backward():
    GPIO.output(motor1A, False)
    GPIO.output(motor1B, True)
    pwm1.ChangeDutyCycle(70)
    GPIO.output(motor2A, False)
    GPIO.output(motor2B, True)
    pwm2.ChangeDutyCycle(78)
   
def right(left, right):
    GPIO.output(motor1A, True)
    GPIO.output(motor1B, False)
    pwm1.ChangeDutyCycle(40)
    GPIO.output(motor2A, False)
    GPIO.output(motor2B, True)
    pwm2.ChangeDutyCycle(0)
        
def left(left, right):
    GPIO.output(motor1A, False)
    GPIO.output(motor1B, True)
    pwm1.ChangeDutyCycle(0)
    GPIO.output(motor2A, True)
    GPIO.output(motor2B, False)
    pwm2.ChangeDutyCycle(40)
    
def stop():
    GPIO.output(motor1A, False)
    GPIO.output(motor1B, False)
    GPIO.output(motor2A, False)
    GPIO.output(motor2B, False)

def stop5():
    GPIO.output(motor1A, False)
    GPIO.output(motor1B, False)
    GPIO.output(motor2A, False)
    GPIO.output(motor2B, False)
    time.sleep(5)

# def uturn():
#     GPIO.output(motor1A, False)
#     GPIO.output(motor1B, True)
#     pwm1.ChangeDutyCycle(50)
#     GPIO.output(motor2A, True)
#     GPIO.output(motor2B, False)
#     pwm2.ChangeDutyCycle(50)
#     time.sleep(0.5)

# def detectstop():
#     for (x,y,w,h) in stopsign:
#         print(w,h)

#         if(w >35):
#             # cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
#             # cv2.putText(image, 'STOP', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             stop5()
#             t_end = time.time()
            
#             while(True):
#                 set_path1(masked_image, 120)
#                 if (time.time() > t_end + 8):
#                     break
#                 break
#             break
                
            

            # flag += 1
            # print(flag)
        
 #   Timer(5, detectstop).start()

motor1A = 8
motor1B = 10
motor2A = 3
motor2B = 5
p1 = 12
p2 = 7


GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings (False)
GPIO.setup(motor1A, GPIO.OUT)
GPIO.setup(motor1B, GPIO.OUT)
GPIO.setup(motor2A, GPIO.OUT)
GPIO.setup(motor2B, GPIO.OUT)

GPIO.setup(p1, GPIO.OUT)
GPIO.setup(p2, GPIO.OUT)
pwm1 = GPIO.PWM(p1,100)
pwm2 = GPIO.PWM(p2,100)
pwm1.start(0)
pwm2.start(0)


camera = PiCamera()
camera.resolution = (320, 224)
camera.framerate = 15
camera.vflip = True
camera.hflip = True
rawCapture = PiRGBArray(camera, size=(320,224))
time.sleep(.1)

# prev_dist = 5

# flag = 0
# camera.start_recording('video.h264') # start recording


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):    
    
    image = frame.array
    
    # cv2.imshow('image', image)

    detect_marker()

    real = image

    masked_image = select_white(real,140)
    
    height, width = masked_image.shape
    center=int(width/2)
    # uturn_dist = min(int(height),int(first_nonzero(masked_image[:,center],0,height))-1)
    # if uturn_dist < 5:
    #     uturn()
    

    # stopsign = stopxml.detectMultiScale(image, 
    #     scaleFactor=1.1,
    #     minNeighbors=15,
    #     minSize=(10,10),
    #     maxSize=(60,60)
    #     )
    
    # detectstop()
        
    # dist = measure()    
    
    # if dist < 1:
    #     dist = prev_dist
    # prev_dist = dist

    a=set_path1(masked_image,130)
    b = a[0]
      
    # if dist < 15:
    #     b = 'stop5'
    #     detect_marker_stop()
        

    print(b, a[1], a[2], a[3])# '|' , dist)    
    
    if b == 'forward':
        forward(a[1], a[2])
    elif b == 'left':
        left(a[1],a[2])
    elif b == 'right':
        right(a[1],a[2])
    elif b == 'backward':
        backward()
    else:
        stop()
    
                        
    ##
    # cv2.imshow("real", real)
    cv2.imshow("Masked", masked_image)
    
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)

    if key == ord('q'):
        break
# camera.stop_recording()