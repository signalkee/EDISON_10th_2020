# 10th EDISON Computation Design Challange TEAM ROBOIN 1
# I.H. KEE, Y.W. lEE, I.W. OH
import math
import RPi.GPIO as GPIO
import cv2
import numpy as np
import time
#from timer import Timer
from ar_markers import detect_markers
from skimage.measure import block_reduce
from picamera import PiCamera
from picamera.array import PiRGBArray

global flag


# =====================================================================================
# =================================    DETECTING    ===================================
# =====================================================================================

# Path of train weights and lables 
labelsPath = "./edison.names"
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = "2nd_final.weights"
configPath = "./edison.cfg"

# Loading the neural network
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)

# Detecting stopsign
def detect_stop(image):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    (H, W) = image.shape[:2]
    
    # determine only the "ouput" layers name which we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # construct a blob from the input image and then perform a forward pass of the YOLO object detector, 
    # giving us our bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (320, 224), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    confidences = []
    classIDs = []
    threshold = 0.1 # parameter
    
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            # confidence type=float, default=0.5
            
#             text = 'labels: '+ str(LABELS[classID])+'//'+ str(confidence)
#             cv2.putText(image, text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)
#             print('/////////////////conf: {}'.format(confidence))
            
            if confidence > threshold:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)
#     if confidences[0] is not None:
    print('/////////////////conf: {}'.format(confidences))
    # ensure at least one detection exists
    if len(idxs) > 0:
        # flag = 0 for stopsign detection
        print(width)
        global flag
        global stop_flag
        flag =+ 1
        stop_flag =+ 1
        print('stop detected')
        # print(flag)
        # if stopsign detected, stop for 5secs
        stop5()
        #stop_flag += 1 
#         stop_time = time.time()
        # loop over the indexes we are keeping 
        # for i in idxs.flatten():
        #     # extract the bounding box coordinates
        #     (x, y) = (boxes[i][0], boxes[i][1])
        #     (w, h) = (boxes[i][2], boxes[i][3])

#          draw a bounding box rectangle and label on the image
#             color = (255,0,0)
#             cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
#             text = 'labels: '+ str(ABELS[classIDs[i]])+'//'+ str(confidences[i])
#             cv2.putText(image, text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)
    
    return image

# Detecting AR Markers
def detect_marker(forward):
    # flag################################################################## 
    markers = detect_markers(realar)
    # if markers is True:
    #     flag =+ 1
    for marker in markers:
        for pos in marker.contours:
            global flag
            global pos1
            pos1.append(pos)
        diff1 = pos1[0]-pos1[1]
        diff2 = pos1[1]-pos1[2]
        ref1 = max([abs(diff1[0][0]), abs(diff1[0][1])])
        ref2 = max([abs(diff2[0][0]), abs(diff2[0][1])])
        print('area: {}'.format(ref1*ref2))
        
        if marker.id == 114: #left
            print('//////////////////////////sibalnoma')
            if ref1*ref2 > 2000:
                print('///////////////////ARMARKER_DETECTED!!!!////////     ID: {}'.format(marker.id)) 
                
    #             time.sleep(0.7)
                flag = 2
                GPIO.output(motor1A, False)
                GPIO.output(motor1B, True)
                pwm1.ChangeDutyCycle(0)
                GPIO.output(motor2A, True)
                GPIO.output(motor2B, False)
                pwm2.ChangeDutyCycle(50)
                time.sleep(0.4)

        elif marker.id == 1156: #right
            print('//////////////////////////sibalnoma')
            if ref1*ref2 > 2000 :
                print('///////////////////ARMARKER_DETECTED!!!!////////     ID: {}'.format(marker.id))
    #                     for pos in marker.contours:
    #                         print(pos)
    #             time.sleep(0.7)
                flag = 2
                GPIO.output(motor1A, True)
                GPIO.output(motor1B, False)
                pwm1.ChangeDutyCycle(50)
                GPIO.output(motor2A, False)
                GPIO.output(motor2B, True)
                pwm2.ChangeDutyCycle(0)
                time.sleep(0.4)


# =====================================================================================
# ==============================    IMAGE PROCESSING    ===============================
# =====================================================================================
# detect first nonzero pixel H, V
def first_nonzero(arr, axis, invalid_val=-1):
    arr = np.flipud(arr)
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

# detect red
def red_image(image):
    red_detection = 'no_red'
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
    lower_range = np.array([150,150,0], dtype=np.uint8)
    upper_range = np.array([180,255,255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    min_pool=block_reduce(mask, block_size=(2,2), func=np.min)
    
    return min_pool

def crop(img,dx,dy):
    y,x,z = img.shape
    startx = x//2-(dx)
    starty = y-(dy)
    return img[starty:y,startx:startx+2*dx]
    
# change white & black with threshold
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
        if left_sum/right_sum > 1.354: 
            result = 'left'
        elif right_sum/left_sum > 1.325:
            result = 'right'
        elif forward_sum > 260:
            result = 'forward'
        elif forward_sum > 100: 
            if left_sum/right_sum > 1.177:
                result = 'left'
            elif right_sum/left_sum > 1.273:
                result = 'right'
            else:
                result = 'forward'
        else: 
            result = 'backward'
#     
    if flag == 1:
        if left_sum/right_sum > 1.442: 
                result = 'left'
        elif right_sum/left_sum > 1.325:
            result = 'right'
        elif forward_sum > 300: 
            result = 'forward'
        elif forward_sum > 100: 
            if left_sum/right_sum > 1.166: 
                result = 'left'
            elif right_sum/left_sum > 1.097:
                result = 'right'
            else:
                result = 'forward'
        else: 
            result = 'backward'
    if flag == 2:
        if left_sum > right_sum + 600: 
                result = 'left'
        elif left_sum < right_sum - 600:
            result = 'right'
        elif forward_sum > 30: 
            result = 'forward'
#         elif forward_sum > 100: 
#             if left_sum > right_sum + 100: 
#                 result = 'left'
#             elif left_sum < right_sum - 100:
#                 result = 'right'
#             else:
#                 result = 'forward'
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
    global KI
    global KD
    global limit
    global output_min

    if result == 'left':
        err = left_sum - right_sum
    elif result == 'right':
        err = right_sum - left_sum
    else:
        err = 0
        
    P_err = KP * err
    # I_err += KI * ((err + pre_err) / 2) * LOOPTIME
    # D_err = KD * (err - pre_err) / LOOPTIME
    # if (I_err > I_err_limit) I_err = I_err_limit
    # else if (I_err < -I_err_limit) I_err = -I_err_limit
    # output = abs(P_err + I_err + D_err)
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
    pwm1.ChangeDutyCycle(50)
    GPIO.output(motor2A, False)
    GPIO.output(motor2B, True)
    pwm2.ChangeDutyCycle(50)
   
def right(fullspeed, ctrl):
    GPIO.output(motor1A, True)
    GPIO.output(motor1B, False)
    pwm1.ChangeDutyCycle(fullspeed)
    GPIO.output(motor2A, False)
    GPIO.output(motor2B, True)
    # pwm2.ChangeDutyCycle(0)
    pwm2.ChangeDutyCycle(fullspeed-ctrl)
        
def left(fullspeed, ctrl):
    GPIO.output(motor1A, False)
    GPIO.output(motor1B, True)
    # pwm1.ChangeDutyCycle(0)

    pwm1.ChangeDutyCycle(fullspeed-ctrl)
    GPIO.output(motor2A, True)
    GPIO.output(motor2B, False)
    pwm2.ChangeDutyCycle(fullspeed)
    
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

def parking():
    GPIO.output(motor1A, True)
    GPIO.output(motor1B, False)
    pwm1.ChangeDutyCycle(40) #left
    GPIO.output(motor2A, False)
    GPIO.output(motor2B, True)
    pwm2.ChangeDutyCycle(40) #right
    time.sleep(1.25)

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
fullspeed = 60
# ===  control variables  ===
global KP
global KI
global KD
global limit
global output_min

KP = 0.15
KI = 0
KD = 0
limit = 35
output_min = 3
# LOOPTIME = 1/15


camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 45
camera.hflip = True
camera.vflip = True
rawCapture = PiRGBArray(camera, size=(320,240))
# camera.start_recording('video.h264') # start recording
time.sleep(.1)

'''
flag = 0 for just driving 
flag = 1 for stopsign detection
flag = 2 for ar marker detection
'''
flag = 0
# stop_flag = 0
# start_time = time.time()

# =====================================================================================
# =============================          MAIN           ===============================
# =====================================================================================
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):    
    st0 = time.time()
    image = frame.array
    
    real = crop(image,160,120)
    realar = crop(image, 160, 180)
#     if flag == 0:
    
#     st1 = time.time()
#     print('get_image: {}'.format(st1-st0))
    red = red_image(real)
#     st2 = time.time()
#     print('red_image: {}'.format(st2-st1))
    redcnt = np.count_nonzero(red)
    print('how_many_redcnt: {}'.format(redcnt))
#     st3 = time.time()
#     print('redcnt: {}'.format(st3-st2))
    #print('redcnt: {}'.format(redcnt))
    #print('flag: {}'.format(flag))
    pos1=[]
    if redcnt >= 9 :
        if flag == 0:
            detect_stop(image)
#     st4 = time.time()
#     print('detect_stop: {}'.format(st4-st3))
        
    
#     st5 = time.time()
#     print('mrk: {}'.format(st5-st4))

    
    masked_image = select_white(real,140)
    height, width = masked_image.shape
    center=int(width/2)
    
#     st6 = time.time()
#     print('masked+height+center: {}'.format(st6-st5))
  
    a=set_path1(masked_image,120)
#     st7 = time.time()
#     print('set_path: {}'.format(st7-st6))
    if flag == 1:
        detect_marker(a[1])
    
    b = a[0]
    
    ctrl_output = ctrl(b, a[1], a[2], a[3])
#     st8 = time.time()
#     print('ctrl: {}'.format(st8-st7))
    print(b, a[1], a[2], a[3], 'redcnt: {}'.format(redcnt),'ctrl_output: {}'.format(ctrl_output), 'flag: {}'.format(flag))    

    # if b == 'forward':
    #     forward(fullspeed)
    # elif b == 'left':
    #     left(fullspeed-10,0)
    # elif b == 'right':
    #     right(fullspeed-10,0)
    # elif b == 'backward':
    #     backward()
    # else:
    #     stop()

    # P control added
    if b == 'forward':
        forward(fullspeed)
    elif b == 'left':
        left(fullspeed-3, fullspeed-ctrl_output)
    elif b == 'right':
        right(fullspeed-3, ctrl_output)
    elif b == 'backward':
        backward()
    elif b == 'stop':
        forward(fullspeed-10)
        time.sleep(0.5)
        parking()
        stop5()
        
#     st9 = time.time()
#     print('decision: {}'.format(st9-st8))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = 'decision ' + str(b)
    cv2.putText(masked_image, text1, (10,20), font, 0.5, (255, 255, 255), 1)
    
    
                        
    cv2.imshow('image', image)
#     cv2.imshow("Masked", masked_image)
    cv2.imshow('red', red)
    st10 = time.time()
#     print('imshow: {}'.format(st10-st9))
    print('total: {}'.format(st10-st0))
    
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord('q'):
        break

# camera.stop_recording()
