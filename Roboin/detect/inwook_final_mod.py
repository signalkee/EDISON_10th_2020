# 10th EDISON Computation Design Challange TEAM ROBOIN 1
# I. KEE, Y.W. lEE, I.W. OH

import RPi.GPIO as GPIO
import cv2
import numpy as np
import time
# from timer import Timer
from ar_markers import detect_markers

from picamera import PiCamera
from picamera.array import PiRGBArray

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
    threshold = 0.9 # parameter
    
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

    # ensure at least one detection exists
    if len(idxs) > 0:
        # flag = 0 for stopsign detection
        global flag 
        flag =+ 1
        # print(flag)
        # if stopsign detected, stop for 5secs
        stop5()
        # loop over the indexes we are keeping 
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = (255,0,0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x +15, y - 10), cv2.FONT_HERSHEY_SIMPLEX,1, color, 2)
    
    # Timer(5, detect_stop).start() #detect stop sign every 5 seconds
            
            
            
    return image

# Detecting AR Markers
def detect_marker():
    # flag################################################################## 
    markers = detect_markers(image)
    for marker in markers:
        # for pos[[x1,y1]] in marker.contours:
        #     print(x1, y1)
                    
        if marker.id == 114: #left
            print(marker.id)
            time.sleep(0.7)
            GPIO.output(motor1A, False)
            GPIO.output(motor1B, True)
            pwm1.ChangeDutyCycle(0)
            GPIO.output(motor2A, True)
            GPIO.output(motor2B, False)
            pwm2.ChangeDutyCycle(30)
            time.sleep(1)

        elif marker.id == 1156: #right
            print(marker.id)
            time.sleep(0.7)
            GPIO.output(motor1A, True)
            GPIO.output(motor1B, False)
            pwm1.ChangeDutyCycle(30)
            GPIO.output(motor2A, False)
            GPIO.output(motor2B, True)
            pwm2.ChangeDutyCycle(0)
            time.sleep(0.8)

# =====================================================================================
# ==============================    IMAGE PROCESSING    ===============================
# =====================================================================================

# detect first nonzero pixel H, V
def first_nonzero(arr, axis, invalid_val=-1):
    arr = np.flipud(arr)
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

# change white & black with threshold
def select_white(image, white):
   lower = np.array([white,white,white])
   upper = np.array([255,255,255])
   white_mask = cv2.inRange(image, lower, upper)
   return white_mask

# integration path algorithm, returns final action and integrated numbers
def set_path1(image, upper_limit, fixed_center = 'False'):
    height, width = image.shape # shape of array ex) 224,320
    height = height-1 # array starts from 0, so the last num is 319, not 320
    width = width-1
    center=int(width/2)
    left=0
    right=width

    # for integration of left, right road
    white_distance = np.zeros(width)
    delta_w = 10 # parameter
    delta_h = 3  # parameter
    
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
    
    left_sum = np.sum(white_distance[left:center])
    right_sum = np.sum(white_distance[center:right])
    forward_sum = np.sum(white_distance[center-10:center+10])
    
    if left_sum > right_sum + 500: 
            result == 'left'
    elif left_sum < right_sum - 500:
        result = 'right'
    elif forward_sum > 300: 
        result = 'forward'
    elif forward_sum > 100: 
        if left_sum > right_sum + 100: 
            result == 'left'
        elif left_sum < right_sum - 100:
            result == 'right'
        else:
            result = 'forward'
    else: 
        result = 'backward'
    
    return result, forward_sum, left_sum, right_sum


# =====================================================================================
# ==============================     MOTOR CONTROL      ===============================
# =====================================================================================

# P control -> constrained P control mod. needed
# output is amount of dutycycle to reduce -> (fullspeed - output) is needed
# def ctrl(result, forward_sum, left_sum, right_sum): 
#     if result == 'left':
# 	    err = left_sum - right_sum
#     elif result == 'right':
#         err = right_sum - left_sum
#     else:
#         err = 0
    
# 	P_err = KP * err
# 	# I_err += KI * ((err + pre_err) / 2) * LOOPTIME
# 	# D_err = KD * (err - pre_err) / LOOPTIME
# 	# if (I_err > I_err_limit) I_err = I_err_limit
# 	# else if (I_err < -I_err_limit) I_err = -I_err_limit
# 	# output = abs(P_err + I_err + D_err)
#    	output = abs(P_err)

# 	if output >= limit:
#         output = limit
# 	if output < output_min:
#         output = output_min

#     return output

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
    pwm2.ChangeDutyCycle(fullspeed-ctrl)
        
def left(fullspeed, ctrl):
    GPIO.output(motor1A, False)
    GPIO.output(motor1B, True)
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



# =====================================================================================
# ==============================         SETUP          ===============================
# =====================================================================================
motor1A = 8
motor1B = 10
motor2A = 5
motor2B = 3
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
fullspeed = 50 ###############################################################

# ===  control variables  ===
# KP = 0.05
# KI = 0
# KD = 0
# limit = 35
# output_min = 5
# LOOPTIME = 1/15

camera = PiCamera()
camera.resolution = (320, 224)
camera.framerate = 15
rawCapture = PiRGBArray(camera, size=(320,224))
camera.start_recording('video.h264') # start recording
time.sleep(.1)

flag = 0 
'''
flag = 0 for just driving 
flag = 1 for stopsign detection
flag = 2 for ar marker detection
'''
start_time = time.time()
# =====================================================================================
# =============================          MAIN           ===============================
# =====================================================================================
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):    

    image = frame.array
    masked_image = select_white(image,140) # parameter 
   
    a=set_path1(masked_image,130) # parameter 
    b = a[0]
    
    # ctrl_output = ctrl(b, a[1], a[2], a[3])
    # print(b, a[1], a[2], a[3], ctrl_output)    
    print(b, a[1], a[2], a[3]) 
    
    if b == 'forward':
        forward(fullspeed)
    elif b == 'left':
        left(fullspeed-10,0)
    elif b == 'right':
        right(fullspeed-10,0)
    elif b == 'backward':
        backward()
    else:
        stop()

    # After 7secs, start detecting stop sign
    if time.time() - start_time > 7:
        flag = 1

    if flag == 1:
        detect_markers
    # P control added
    # if b == 'forward':
    #     forward(fullspeed)
    # elif b == 'left':
    #     left(fullspeed-10, ctrl_output)
    # elif b == 'right':
    #     right(fullspeed-10, ctrl_output)
    # elif b == 'backward':
    #     backward()
    # else:
    #     stop()
    
    # Visualization                
    # cv2.imshow('image', image)
    # cv2.imshow('Stopsign', 'Predict_image')
    # cv2.imshow("Masked", masked_image)
    
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord('q'):
        break

# camera.stop_recording()