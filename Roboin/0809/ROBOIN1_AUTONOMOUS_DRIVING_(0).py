# 10th EDISON Computation Design Challange TEAM ROBOIN 1
# I.H. KEE, Y.W. lEE, I.W. OH

import math
import RPi.GPIO as GPIO
import cv2
import numpy as np
import time
from ar_markers import detect_markers
from skimage.measure import block_reduce
from picamera import PiCamera
from picamera.array import PiRGBArray

global flag

# =====================================================================================
# =================================    DETECTING    ===================================
# =====================================================================================

# ================================     TINY-YOLO    ===================================
# Path of train weights and lables 
labelsPath = "./edison.names"
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = "2nd_final.weights"
configPath = "./edison.cfg"

# Loading the neural network
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)

# Detecting stopsign
def detect_stop(image):
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    (H, W) = image.shape[:2]
    
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (320, 224), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    confidences = []
    classIDs = []
    threshold = 0.1 # parameter
    

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)

    if len(idxs) > 0:
        global flag
        flag =+ 1
        print('stop detected')
        stop5()

    return image

# Detecting AR Markers
def detect_marker(realar,forward):
    markers = detect_markers(realar)

    for marker in markers:
        global flag

        for pos in marker.contours:
            global pos1
            pos1.append(pos)
        diff1 = pos1[0]-pos1[1]
        diff2 = pos1[1]-pos1[2]
        ref1 = max([abs(diff1[0][0]), abs(diff1[0][1])])
        ref2 = max([abs(diff2[0][0]), abs(diff2[0][1])])
        print('area: {}'.format(ref1*ref2))
        
        if marker.id == 114: #left
            print('//////////////////////////ID_DETECTED')
            flag = 5
            if ref1*ref2 > 1100:
                print('///////////////////+AREA+ARMARKER_DETECTED!!!!////////     ID: {}'.format(marker.id)) 
                flag = 3
                GPIO.output(motor1A, False)
                GPIO.output(motor1B, True)
                pwm1.ChangeDutyCycle(0)
                GPIO.output(motor2A, True)
                GPIO.output(motor2B, False)
                pwm2.ChangeDutyCycle(60)
                time.sleep(0.30)

        elif marker.id == 1156: #right
            print('//////////////////////////ID_DETECTED')
            flag = 5
            if ref1*ref2 > 1100 :
                print('///////////////////+AREA+ARMARKER_DETECTED!!!!////////     ID: {}'.format(marker.id))
                flag = 2
                GPIO.output(motor1A, True)
                GPIO.output(motor1B, False)
                pwm1.ChangeDutyCycle(60)
                GPIO.output(motor2A, False)
                GPIO.output(motor2B, True)
                pwm2.ChangeDutyCycle(0)
                time.sleep(0.30)


# =====================================================================================
# ==============================    IMAGE PROCESSING    ===============================
# =====================================================================================

# detect red & minpooled
def red_image(image):
    red_detection = 'no_red'
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
    lower_range = np.array([150,150,0], dtype=np.uint8)
    upper_range = np.array([180,255,255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    min_pool=block_reduce(mask, block_size=(3,3), func=np.min)
    
    return min_pool

# crop the image
def crop(img,dx,dy):
    y,x,z = img.shape
    startx = x//2-(dx)
    starty = y-(dy)
    return img[starty:y,startx:startx+2*dx]
    
# change image binary with threshold
def select_white(image, white):
   lower = np.array([white,white,white])
   upper = np.array([255,255,255])
   white_mask = cv2.inRange(image, lower, upper)
   return white_mask

# integration path algorithm, returns final action and integrated numbers
def set_path1(image, upper_limit, fixed_center = 'False'):
    height, width = image.shape # shape of array ex) 240,320
    height = height-1 # array starts from 0, so the last num is 319, not 320
    width = width-1
    center=int(width/2)
    left=0
    right=width

    # for integration of left, right road
    white_distance = np.zeros(width)
    delta_w = 8
    delta_h = 3  
    
    if not fixed_center: 
        #finding first white pixel in the lowest row and reconfiguring center pixel position
        for i in range(center):
            if image[height,center-i] > 200:
                left = center-i
                break            
        for i in range(center):
            if image[height,center+i] > 200:
                right = center+i
                break    
        center = int((left+right)/2)      

    # integrating area of left, right road
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
    
    left_sum = np.sum(white_distance[left:center]+1)
    right_sum = np.sum(white_distance[center:right]) 
    forward_sum = np.sum(white_distance[center-10:center+10])
    
    if flag == 0:
        if left_sum > right_sum + 600: 
            result = 'left'
        elif left_sum < right_sum - 600:
            result = 'right'
        elif forward_sum > 400:
            result = 'forward'
        elif forward_sum > 100: 
            if left_sum > right_sum + 73:
                result = 'left'
            elif left_sum < right_sum - 73:
                result = 'right'
            else:
                result = 'forward'
        else: 
            result = 'backward'
  
    if flag == 1 or flag == 5:
        if left_sum > right_sum + 600: 
                result = 'left'
        elif left_sum < right_sum - 600:
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
        else: 
            result = 'backward'
    if flag == 2: #right
        if forward_sum <= 30 :
            result = 'stop'
        elif left_sum > right_sum + 20: 
                result = 'left'
        elif left_sum < right_sum - 10:
            result = 'right'
        elif forward_sum > 30: 
            result = 'forward'
        else:
            result = 'stop'
    if flag == 3: #left
        if forward_sum <= 30 :
            result = 'stop'
        elif left_sum > right_sum + 20: 
                result = 'left'
        elif left_sum < right_sum - 20:
            result = 'right'
        elif forward_sum > 30: 
            result = 'forward'
        else:
            result = 'stop'

    return result, forward_sum, left_sum, right_sum


# =====================================================================================
# ==============================     MOTOR CONTROL      ===============================
# =====================================================================================

# P control -> constrained P control mod. needed
# output is amount of dutycycle to reduce -> (fullspeed - output) is needed
def ctrl(result, forward_sum, left_sum, right_sum):  

    global KP
    global limit
    global output_min

    if result == 'left':
        err = left_sum - right_sum
    elif result == 'right':
        err = right_sum - left_sum
    else:
        err = 0
        
    P_err = KP * err
    output = abs(P_err)

    if output >= limit :
        output = limit
    if output < output_min:
        output = output_min

    return output

def forward(fullspeed):  
    GPIO.output(motor1A, True)
    GPIO.output(motor1B, False)
    pwm1.ChangeDutyCycle(fullspeed) #left
    GPIO.output(motor2A, True)
    GPIO.output(motor2B, False)
    pwm2.ChangeDutyCycle(fullspeed) #right

def backward():
    GPIO.output(motor1A, False)
    GPIO.output(motor1B, True)
    pwm1.ChangeDutyCycle(35)
    GPIO.output(motor2A, False)
    GPIO.output(motor2B, True)
    pwm2.ChangeDutyCycle(35)
   
def right(fullspeed, ctrl):
    GPIO.output(motor1A, True)
    GPIO.output(motor1B, False)
    pwm1.ChangeDutyCycle(fullspeed+ctrl+20)
    GPIO.output(motor2A, True)
    GPIO.output(motor2B, False)
    pwm2.ChangeDutyCycle(fullspeed-ctrl-5)

def right1(fullspeed,ctrl):
    GPIO.output(motor1A, True)
    GPIO.output(motor1B, False)
    pwm1.ChangeDutyCycle(fullspeed+ctrl)
    GPIO.output(motor2A, False)
    GPIO.output(motor2B, False)
    pwm2.ChangeDutyCycle(0)    
    
def right2(fullspeed):
    GPIO.output(motor1A, True)
    GPIO.output(motor1B, False)
    pwm1.ChangeDutyCycle(fullspeed)
    GPIO.output(motor2A, False)
    GPIO.output(motor2B, True)
    pwm2.ChangeDutyCycle(fullspeed)
        
def left(fullspeed, ctrl):
    GPIO.output(motor1A, True)
    GPIO.output(motor1B, False)
    pwm1.ChangeDutyCycle(fullspeed-ctrl-5)
    GPIO.output(motor2A, True)
    GPIO.output(motor2B, False)
    pwm2.ChangeDutyCycle(fullspeed+ctrl+20)
    
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

# performance!!
def parking():
    GPIO.output(motor1A, True)
    GPIO.output(motor1B, False)
    pwm1.ChangeDutyCycle(40) #left
    GPIO.output(motor2A, False)
    GPIO.output(motor2B, True)
    pwm2.ChangeDutyCycle(40) #right
    time.sleep(1.4)

# =====================================================================================
# ==============================         SETUP          ===============================
# =====================================================================================

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
fullspeed = 45

# ===  control variables  ===
global KP
global limit
global output_min

KP = 0.05
limit = 20
output_min = 3
flag = 0

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 45
camera.hflip = True
camera.vflip = True
rawCapture = PiRGBArray(camera, size=(320,240))
time.sleep(.1)

'''
flag = 0 before stopsign
flag = 1 after stopsign & before ARmarker
flag = 2 ater ARmarker
'''

# =====================================================================================
# =============================          MAIN           ===============================
# =====================================================================================

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):    
    st0 = time.time()
    image = frame.array
    
    real = crop(image,160,120)

    pos1=[]
    if flag ==0 :
        realred = crop(image,160,140)
        red = red_image(realred)
        redcnt = np.count_nonzero(red)
        print('how_many_redcnt: {}'.format(redcnt))
        if redcnt >= 5 :
            detect_stop(image)
    
    masked_image = select_white(real,140)
    height, width = masked_image.shape
    center=int(width/2)
    
    a=set_path1(masked_image,120)

    if flag == 1 or flag ==5 :
        realar = crop(image, 160, 180)
        detect_marker(realar,a[1])
    
    b = a[0]
    
    ctrl_output = ctrl(b, a[1], a[2], a[3])

    print(b, a[1], a[2], a[3],'ctrl_output: {}'.format(ctrl_output), 'flag: {}'.format(flag))    

    if flag == 5:
        fullspeed = 35
    else:
        fullspeed = 45

    # P control added
    if b == 'forward':
        forward(fullspeed)
    elif b == 'left':
        left(fullspeed-0.15*limit, 0.9*ctrl_output)
    elif b == 'right':
        right(fullspeed-0.35*limit,ctrl_output)
        if flag ==1:
            right1(fullspeed,0.5*ctrl_output)
    elif b == 'backward':
        backward()
        if flag == 1 or flag ==5:
            right2(60)
    elif b == 'stop':
        forward(fullspeed-10)
        time.sleep(0.5)
        parking()
        stop5()
        break
        
    # add_info. to masked_image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = 'decision ' + str(b)
    cv2.putText(masked_image, text1, (10,20), font, 0.5, (255, 255, 255), 1)

    # all kinds of imshow                  
#     cv2.imshow('image', image)
#     cv2.imshow("Masked", masked_image)
#     cv2.imshow('red', red)

    st10 = time.time()
    print('total: {}'.format(st10-st0))
    
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord('q'):
        break