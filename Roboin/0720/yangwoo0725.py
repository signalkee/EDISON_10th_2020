import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import math
#import RPi.GPIO as GPIO
import picamera
import time
#from ar_markers import detect_markers
from picamera.array import PiRGBArray
import sys
import linecache

# ============================================================@@ FUNCTIONS @@===================================================================

# @@ DETECTION @@ ====================================================================================================
def first_nonzero(arr, axis, invalid_val=-1):
    arr = np.flipud(arr)
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def select_white(image, white):
    lower = np.uint8([white,white,white])
    upper = np.uint8([255,255,255])
    white_mask = cv2.inRange(image, lower, upper)
    return white_mask

def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    max_value = np.max(abs_sobel)
    binary_output = np.uint8(255*abs_sobel/max_value)
    threshold_mask = np.zeros_like(binary_output)
    threshold_mask[(binary_output >= thresh_min) & (binary_output <= thresh_max)] = 1
    return threshold_mask

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobel_y,abs_sobel_x)
    direction = np.absolute(direction)
    # 5) Create a binary mask where direction thresholds are met
    mask = np.zeros_like(direction)
    mask[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return mask

def color_grid_thresh(img, s_thresh=(170,255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivateive in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivateive to accentuate lines
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # combine the two binary
    binary = sxbinary | s_binary

    # Stack each channel (for visual check the pixal sourse)
    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary,s_binary)) * 255
    return binary

def get_thresholded_image(img):
    
    #img = cv2.undistort(img, cameraMatrix, distortionCoeffs, None, cameraMatrix)
    
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    height, width = gray.shape
    
    # apply gradient threshold on the horizontal gradient
    sx_binary = abs_sobel_thresh(gray, 'x', 10, 200)

    # apply gradient direction threshold so that only edges closer to vertical are detected.
    dir_binary = dir_threshold(gray, thresh=(np.pi/6, np.pi/2))
    
    # combine the gradient and direction thresholds.
    combined_condition = ((sx_binary == 1) & (dir_binary == 1))

    # R & G thresholds so that yellow lanes are detected well.
    color_threshold = 150
    R = img[:,:,0]
    G = img[:,:,1]
    color_combined = np.zeros_like(R)
    r_g_condition = (R > color_threshold) & (G > color_threshold)
    
    # color channel thresholds
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    L = hls[:,:,1]
    
    # S channel performs well for detecting bright yellow and white lanes
    s_thresh = (50, 255)
    s_condition = (S <= s_thresh[1]) & (S > s_thresh[0]) #for bright yellow
    
    # We put a threshold on the L channel to avoid pixels which have shadows and as a result darker.
    l_thresh = (100, 255)
    l_condition = (L > l_thresh[0]) & (L <= l_thresh[1])

    # combine all the thresholds
    # A pixel should either be a yellowish or whiteish
    # And it should also have a gradient, as per our thresholds
    color_combined[(r_g_condition & l_condition) & (s_condition | combined_condition)] = 255 #- with yellow detect
    #color_combined[(l_condition) & (s_condition | combined_condition)] = 1
    
    # apply the region of interest mask
    mask = np.zeros_like(color_combined)
    region_of_interest_vertices = np.array([[1,height-1],[1,180], [int(width/3), 140],[int(width*2/3), 140], [width-1, 180], [width-1,height-1]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], (255,255,255))
    thresholded = cv2.bitwise_and(color_combined, mask)
    
    return thresholded #-for + ROI 
    #return color_combined

# def detect_marker():
#     markers = detect_markers(image)
#     for marker in markers:
#         if marker.id == 114: #left - change
#             print("ARmarker id:  %d  ==> To the left" % marker.id)
#             time.sleep(0.7)
#             #motor(-50,50)
#             time.sleep(1)
#         elif marker.id == 228: #right - change
#             print("ARmarker id:  %d  ==> To the right" % marker.id)
#             time.sleep(0.7)
#             #motor(50,-50)
#             time.sleep(1)

# ==============================================================================================================================

# =============================================================================
# img = mpimg.imread('test_images/left.jpg')
# image_shape = img.shape
# thresholded = get_thresholded_image(img)
# =============================================================================
# #img = cv2.undistort(img, cameraMatrix, distortionCoeffs, None, cameraMatrix)
# cv2.imwrite('thresholded.jpg',thresholded)
# 
# # Plot the 2 images side by side
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(thresholded, cmap='gray')
# ax2.set_title('Thresholded Image', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# =============================================================================

# @@ PATHPLANNING @@ ====================================================================================================
# def measure_radius_of_curvature(x_values,num_rows):
#     ym_per_pix = 30/720 # meters per pixel in y dimension
#     xm_per_pix = 3.7/700 # meters per pixel in x dimension
#     # If no pixels were found return None
#     y_points = np.linspace(0, num_rows-1, num_rows)
#     y_eval = np.max(y_points)
    
#     # Fit new polynomials to x,y in world space
#     fit_cr = np.polyfit(y_points*ym_per_pix, x_values*xm_per_pix, 2)
#     curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / (2*fit_cr[0])
#     return curverad

# def set_path(img, image_shape, thresholded):
#     # Vertices extracted manually for performing a perspective transform => please manually modified!!!!!!!!!!
#     bottom_left_before = [50,240]
#     bottom_right_before = [250, 240]
#     top_left_before = [125, 160]
#     top_right_before = [160, 160]
    
#     bottom_left_after = [72,240]
#     bottom_right_after = [200, 240]
#     top_left_after = [72, 1]
#     top_right_after = [200, 1]


#     source = np.float32([bottom_left_before,bottom_right_before,top_right_before,top_left_before])
    
#     pts = np.array([bottom_left_before,bottom_right_before,top_right_before,top_left_before], np.int32)
#     pts = pts.reshape((-1,1,2))
#     # copy = img.copy()
#     # cv2.polylines(copy,[pts],True,(255,0,0), thickness=3)
    
#     # Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
    
    
#     dst = np.float32([bottom_left_after,bottom_right_after,top_right_after,top_left_after])
#     M = cv2.getPerspectiveTransform(source, dst)
#     img_size = (image_shape[1], image_shape[0])
    
#     # Birds eye view
#     warped = cv2.warpPerspective(thresholded, M, img_size , flags=cv2.INTER_LINEAR)
        
#     # =============================================================================
#     # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#     # f.tight_layout()
#     # ax1.imshow(copy)
#     # ax1.set_title('Original Image', fontsize=50)
#     # ax2.imshow(warped, cmap='gray')
#     # ax2.set_title('Warped Image', fontsize=50)
#     # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#     # =============================================================================
    
#     histogram = np.sum(warped[int(warped.shape[0]/2):,:], axis=0)
    
#     # Peak in the first half indicates the likely position of the left lane
#     half_width = np.int(histogram.shape[0]/2)
#     leftx_base = np.argmax(histogram[:half_width])
    
#     # Peak in the second half indicates the likely position of the right lane
#     rightx_base = np.argmax(histogram[half_width:]) + half_width
    
#     out_img = np.dstack((warped, warped, warped))*255
    
#     non_zeros = warped.nonzero()
#     non_zeros_y = non_zeros[0]
#     non_zeros_x = non_zeros[1]
    
#     num_windows = 5
#     num_rows = warped.shape[0]
#     window_height = np.int(num_rows/num_windows)
#     window_half_width = 10
    
#     min_pixels = 20
    
#     left_coordinates = []
#     right_coordinates = []
    
#     for window in range(num_windows):
#         y_max = num_rows - window*window_height
#         y_min = num_rows - (window+1)* window_height
        
#         left_x_min = leftx_base - window_half_width
#         left_x_max = leftx_base + window_half_width
        
#         #cv2.rectangle(out_img, (left_x_min, y_min), (left_x_max, y_max), [0,0,255],2)
        
#         good_left_window_coordinates = ((non_zeros_x >= left_x_min) & (non_zeros_x <= left_x_max) & (non_zeros_y >= y_min) & (non_zeros_y <= y_max)).nonzero()[0]
#         left_coordinates.append(good_left_window_coordinates)
        
#         if len(good_left_window_coordinates) > min_pixels:
#             leftx_base = np.int(np.mean(non_zeros_x[good_left_window_coordinates]))
        
#         right_x_min = rightx_base - window_half_width
#         right_x_max = rightx_base + window_half_width
        
#         #cv2.rectangle(out_img, (right_x_min, y_min), (right_x_max, y_max), [0,0,255],2)
        
#         good_right_window_coordinates = ((non_zeros_x >= right_x_min) & (non_zeros_x <= right_x_max) & (non_zeros_y >= y_min) & (non_zeros_y <= y_max)).nonzero()[0]
#         right_coordinates.append(good_right_window_coordinates)
            
#         if len(good_right_window_coordinates) > min_pixels:
#             rightx_base = np.int(np.mean(non_zeros_x[good_right_window_coordinates]))
            
#     left_coordinates = np.concatenate(left_coordinates)
#     right_coordinates = np.concatenate(right_coordinates)
    
#     out_img[non_zeros_y[left_coordinates], non_zeros_x[left_coordinates]] = [255,0,0]
#     out_img[non_zeros_y[right_coordinates], non_zeros_x[right_coordinates]] = [0,0,255]
    
#     left_x = non_zeros_x[left_coordinates]
#     left_y = non_zeros_y[left_coordinates]
    
#     polyfit_left = np.polyfit(left_y, left_x, 2)
    
#     right_x = non_zeros_x[right_coordinates]
#     right_y = non_zeros_y[right_coordinates]
    
#     polyfit_right = np.polyfit(right_y, right_x, 2)
    
#     y_points = np.linspace(0, num_rows-1, num_rows)
    
#     left_x_predictions = polyfit_left[0]*y_points**2 + polyfit_left[1]*y_points + polyfit_left[2]
    
#     right_x_predictions = polyfit_right[0]*y_points**2 + polyfit_right[1]*y_points + polyfit_right[2]
    
#     # =============================================================================
#     # plt.imshow(out_img)
#     # plt.plot(left_x_predictions, y_points, color='yellow')
#     # plt.plot(right_x_predictions, y_points, color='yellow')
#     # plt.xlim(0, warped.shape[1])
#     # plt.ylim(warped.shape[0],0)
#     # =============================================================================
    
#     # margin = 50
#     # out_img = np.dstack((warped, warped, warped))*255
    
#     # left_x_predictions = polyfit_left[0]*non_zeros_y**2 + polyfit_left[1]*non_zeros_y + polyfit_left[2]
#     # left_coordinates = ((non_zeros_x >= left_x_predictions - margin) & (non_zeros_x <= left_x_predictions + margin)).nonzero()[0]
    
#     # right_x_predictions = polyfit_right[0]*non_zeros_y**2 + polyfit_right[1]*non_zeros_y + polyfit_right[2]
#     # right_coordinates = ((non_zeros_x >= right_x_predictions - margin) & (non_zeros_x <= right_x_predictions + margin)).nonzero()[0]
    
#     # out_img[non_zeros_y[left_coordinates], non_zeros_x[left_coordinates]] = [255,0,0]
#     # out_img[non_zeros_y[right_coordinates], non_zeros_x[right_coordinates]] = [0,0,255]
    
    
#     # left_x = non_zeros_x[left_coordinates]
#     # left_y = non_zeros_y[left_coordinates]
    
#     # polyfit_left = np.polyfit(left_y, left_x, 2)
    
#     # right_x = non_zeros_x[right_coordinates]
#     # right_y = non_zeros_y[right_coordinates]
    
#     # polyfit_right = np.polyfit(right_y, right_x, 2)
    
#     # y_points = np.linspace(0, num_rows-1, num_rows)
    
#     # left_x_predictions = polyfit_left[0]*y_points**2 + polyfit_left[1]*y_points + polyfit_left[2]
    
#     # right_x_predictions = polyfit_right[0]*y_points**2 + polyfit_right[1]*y_points + polyfit_right[2]
#     #=============================================================================================================
#     #window_img = np.zeros_like(out_img)
    
#     #left_line_window_1 = np.array(np.transpose(np.vstack([left_x_predictions - margin, y_points])))
    
#     #left_line_window_2 = np.array(np.flipud(np.transpose(np.vstack([left_x_predictions + margin, y_points]))))
    
#     #left_line_points = np.vstack((left_line_window_1, left_line_window_2))
    
#     #cv2.fillPoly(window_img, np.int_([left_line_points]), [0,255, 0])
    
#     #right_line_window_1 = np.array(np.transpose(np.vstack([right_x_predictions - margin, y_points])))
    
#     #right_line_window_2 = np.array(np.flipud(np.transpose(np.vstack([right_x_predictions + margin, y_points]))))
    
#     #right_line_points = np.vstack((right_line_window_1, right_line_window_2))
    
#     #cv2.fillPoly(window_img, np.int_([right_line_points]), [0,255, 0])
    
#     #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    
#     # =============================================================================
#     # plt.imshow(result)
#     # plt.plot(left_x_predictions, y_points, color='yellow')
#     # plt.plot(right_x_predictions, y_points, color='yellow')
#     # plt.xlim(0, warped.shape[1])
#     # plt.ylim(warped.shape[0],0)
#     # =============================================================================
    
#     # left_curve_rad = measure_radius_of_curvature(left_x_predictions,num_rows)
#     # right_curve_rad = measure_radius_of_curvature(right_x_predictions,num_rows)
#     # average_curve_rad = (left_curve_rad + right_curve_rad)/2
#     # curvature_string = "Radius of curvature: %.2f m" % average_curve_rad
    
#     # compute the offset from the center
#     #lane_center = (right_x_predictions[719] + left_x_predictions[719])/2
#     followpointy = 50 # change to robust value
#     followpointx = (right_x_predictions[followpointy] + left_x_predictions[followpointy])/2
#     # xm_per_pix = 3.7/700 # meters per pixel in x dimension
#     # center_offset_pixels = img_size[0]/2 - lane_center
#     # center_offset_mtrs = xm_per_pix*center_offset_pixels
#     # offset_string = "Center offset: %.2f m" % center_offset_mtrs
        
#     #return average_curve_rad, center_offset_mtrs, followpointx, followpointy
#     return followpointx, followpointy

def set_path(image_shape, thresholded):

    # bottom_left_before = [50,240]
    # bottom_right_before = [250, 240]
    # top_left_before = [125, 160]
    # top_right_before = [160, 160]
    
    # bottom_left_after = [72,240]
    # bottom_right_after = [200, 240]
    # top_left_after = [72, 1]
    # top_right_after = [200, 1]

    # source = np.float32([bottom_left_before,bottom_right_before,top_right_before,top_left_before])
    # #print('66666666666666666666666666666666666')
    # pts = np.array([bottom_left_before,bottom_right_before,top_right_before,top_left_before], np.int32)
    # #print('7777777777777777777777777777777')
    # pts = pts.reshape((-1,1,2))
    # #print('888888888888888888888888888888888888888')
    # # copy = img.copy()
    # # cv2.polylines(copy,[pts],True,(255,0,0), thickness=3)
    
    # # Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
    
    # dst = np.float32([bottom_left_after,bottom_right_after,top_right_after,top_left_after])
    # #print('9999999999999999999999999999999999')
    # M = cv2.getPerspectiveTransform(source, dst)
    # #print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    # img_size = (image_shape[1], image_shape[0])
    
    # # Birds eye view
    # warped = cv2.warpPerspective(thresholded, M, img_size , flags=cv2.INTER_LINEAR)
    #print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
    # print(warped.shape)
    #print(image_shape)
    height,width = image_shape[0],image_shape[1]
    #print('hhhhhhhhhhhhhhhhhhh')
    height = height-1
    #print('jjjjjjjjjjjjjjjjjjjjj')
    width = width-1
    #print('kkkkkkkkkkkkkkkkkkkk')
    center=int(width/2)
    #print('lllllllllllllllllllll')
    left=0
    #print('mmmmmmmmmmmmmmmmmmm')
    right= width
    
    center = int((left+right)/2)
    #print('dfdfffdfdfdffdfddfdfd')
    try:
        if thresholded[height][:center].min(axis=0) == 255:
            left = 0
        else:
            left = thresholded[height][:center].argmin(axis=0)
        if thresholded[height][center:].max(axis=0) ==0:
            right = width
        else:
            right = center+thresholded[height][center:].argmax(axis=0)
        
        center = int((left+right)/2)
        #print('centerrrrr')
        #print(center)
        forward =int(first_nonzero(thresholded[:,center],0,height))-1
        #print('ddddddddddddddddddddddd')
        left_line = first_nonzero(thresholded[height-forward:height,center:],1, width-center)
        right_line = first_nonzero(np.fliplr(thresholded[height-forward:height,:center]),1, center)
        #print("leftline")
        #print(left_line)
        #print("rightline")
        #print(right_line)
        center_y = (np.ones(forward)*2*center-left_line+right_line)/2-center
        center_x = np.vstack((np.arange(forward), np.zeros(forward)))
        #print("center y")
        #print(center_y)
        #print('center x')
        #print(center_x)
        #print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        print("img")
        print(thresholded)

        y = np.nanmean(center_y)
        x = np.nanmean(center_x)
        x,y = int(x), int(y)
        #print(y)
        #print(x)
    except:
        pass
    return x,y
# ==============================================================================================================================

# @@ CONTROL @@=================================================================================================================
def purePursuit(lookAheadPtX, lookAheadPtY):
    ratio_s2w = 1 # steer & wheel angle ratio
    L = 2.845 # front wheel base -> rear wheel base
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

# def curv_offset2vel (curv, offset):
#     fullspeed = 100
#     if abs(curv) > 50000:#change
#         if offset == 0:
#             left = fullspeed
#             right = fullspeed
#         elif offset < 0:
#             left = fullspeed
#             right = fullspeed - offset*500 #change
#         else :
#             left = fullspeed - offset*500 #change
#             right = fullspeed
#     elif -50000 <= curv < 0:
#         if offset == 0:
#             left = fullspeed
#             right = fullspeed
#         elif offset < 0:
#             left = fullspeed
#             right = fullspeed - offset*3000 #change
#         else :
#             left = fullspeed - offset*3000 #change
#             right = fullspeed
#     elif 0 < curv <= 50000:
#         if offset == 0:
#             left = fullspeed
#             right = fullspeed
#         elif offset < 0:
#             left = fullspeed
#             right = fullspeed - offset*3000 #change
#         else :
#             left = fullspeed - offset*3000 #change
#             right = fullspeed

#     return left, right

# def motor(left, right):
#     left = np.clip(left, -100 , 100)
#     right = np.clip(right, -100, 100)
    
#     if left >= 0:
#         left_f = left
#         left_b = 0
#     else:
#         left_f = 0
#         left_b = left
        
#     if right >= 0:
#         right_f = right
#         right_b = 0
#     else:
#         right_f = 0
#         right_b = right
        
#     if left_f > 0 and right_f > 0:
#         p1A.ChangeDutyCycle(left_f)
#         p1B.ChangeDutyCycle(right_f)
#         GPIO.output(motor1A,True)
#         GPIO.output(motor1B,False)
#         GPIO.output(motor2A,False)
#         GPIO.output(motor2B,True)
        
#     elif left_f == 0 and left_b == 0 and right_f == 0 and right_b == 0:
#         p1A.ChangeDutyCycle(left_f)
#         p1B.ChangeDutyCycle(right_f)
#         GPIO.output(motor1A,True)
#         GPIO.output(motor1B,False)
#         GPIO.output(motor2A,False)
#         GPIO.output(motor2B,True)       
        
#     elif left_f > 0 and right_b < 0:
#         p1A.ChangeDutyCycle(left_f)
#         p1B.ChangeDutyCycle(-right_b)
#         GPIO.output(motor1A,True)
#         GPIO.output(motor1B,False)
#         GPIO.output(motor2A,True)
#         GPIO.output(motor2B,False)    
        
#     elif left_b < 0 and right_b <0 :
#         p1A.ChangeDutyCycle(-left_b)
#         p1B.ChangeDutyCycle(-right_b)
#         GPIO.output(motor1A,False)
#         GPIO.output(motor1B,True)
#         GPIO.output(motor2A,True)
#         GPIO.output(motor2B,False)

# ================================================================================================================================

# ============================================================@@ MAIN LOOP @@===================================================================

# @@ SET UP @@=================================================================================================================
# 
# motor1A = 16
# motor1B = 18
# motor2A = 22
# motor2B = 24
# en1 = 40
# en2 = 38
# 
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
# p1A = GPIO.PWM(en1, 1000)
# p1B = GPIO.PWM(en2, 1000)
# 
# p1A.start(0)
# p1B.start(0)
# 
# =============================================================================
camera = picamera.PiCamera()
camera.rotation = 90
camera.resolution = (320, 240)
camera.vflip = True
camera.hflip = True
camera.framerate = 15

rawCapture = PiRGBArray(camera, size=(320,240))

# ================================================================================================================================
#print('11111111111111111111111111')
try:
    #print('2222222222222222222222')
    while True:
        #print('333333333')
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # start_time = time.time()
            img = frame.array
            image_shape = img.shape
            #thresholded = select_white(img,150)
            #thresholded = color_grid_thresh(img)
            thresholded = get_thresholded_image(img)
            #print('44444444444')
        
# =============================================================================
#             if start_time - cas_time > 1:               
#                 gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
#                 mode =detectTUMB(cup_cascade, gray, real)
# =============================================================================

    
# =============================================================================
#             markers = detect_markers(undistort(real))
#             detect=0
#             for marker in markers:
#                 detect = marker.id
#                 marker.highlite_marker(real)
#         
#             ans = ar_markers(detect)
# =============================================================================

            command = set_path(image_shape, thresholded)
            #print('55555555555')
            # curv = command[0]
            # offset = command[1]

            # @@ how to show img to us @@============================================
            # image_show = img
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # text0 = 'curve: ' + str(command[0])
            # text1 = 'offset: ' + str(command[1])
            # cv2.putText(image_show, text0, (200,20), font, 0.5, (255, 0, 0), 1)
            # cv2.putText(image_show, text1, (10,20), font, 0.5, (255, 0, 0), 1)
            # ======================================================================

            lookAheadPtX, lookAheadPtY = command[0], command[1]
            

            # @@ velocity control @@=================================================
            #detect_marker()

            # if lookAheadPtX is not None and lookAheadPtY is not None:
            #     angle = purePursuit(lookAheadPtX, lookAheadPtY)
            #     print("angle = {}  //  left = {} : right = {} ".format(angle,ang2vel(angle)[0],ang2vel(angle)[1])) # -> then fix the motor value
            #     #motor(ang2vel(angle)[0],ang2vel(angle)[1])
            # else :
            #     #motor(curv_offset2vel(curv, offset)[0],curv_offset2vel(curv, offset)[1])
            #     #print("left = {} : right = {} ".format(curv_offset2vel(curv, offset)[0],curv_offset2vel(curv, offset)[1]))
            #     print('backward!!!!!')
            # =======================================================================




# =============================================================================
#             if abs(curv) > 50000: 
#                 if abs(offset) <= 0.03:
#                     #motor(,)
#                 elif offset < -0.03:
#                     #motor(,)
#                 else :
#                     #motor(,)
# =============================================================================

        
            rawCapture.truncate(0)

            #distance = measure()
            
# =============================================================================
#     
#             if obstacle == 'No object':
# 
#                 #if ans == 'forward':                                
#                     if direction == 'forward':
#                         #motor(100,70)
#                         print('forward')
#                     elif direction == 'left':
#                         #motor(54,30)
#                         print('left')
#                     elif direction == 'right':
#                         #motor(45,54)
#                         print('right')
#                     elif direction == 'backward':
#                         #motor(-40,-40)
#                         print('backward')
#                     elif direction == 'lleft':
#                         #motor(45,30)
#                         print('lleft')
#                     elif direction == 'rright':
#                         #motor(30,45)
#                         print('rright')
#                     elif direction == 'llleft':
#                         #motor(53,20)
#                         print('llleft')
#                     elif direction == 'rrright':
#                         #motor(20,53)
#                         print('rrright')
#                     elif direction == 'uturn':
#                         #motor(40,-40)
#                         print('uturn')
# # =============================================================================
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
            # finish_time = time.time() 
            # total_time = finish_time - start_time
            #print(total_time)
            #print('bef')
            detectpoint = cv2.circle(thresholded,(lookAheadPtX, lookAheadPtY),10,(0,0,255),5)
            #print('aft')
            cv2.imshow('analysis',detectpoint)
            rawCapture.truncate(0)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
    cv2.destroyAllWindows()
except:
    pass