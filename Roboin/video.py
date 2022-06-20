from picamera import PiCamera
from picamera.array import PiRGBArray
from time import sleep




camera = PiCamera()
camera.resolution = (320, 224)
camera.framerate = 15
camera.start_preview()
camera.start_recording('recorded.h264')
camera.wait_recording(10)
camera.stop_recording()
camera.stop_preview()