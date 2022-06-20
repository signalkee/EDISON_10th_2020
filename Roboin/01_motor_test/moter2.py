#library
import RPi.GPIO as GPIO
import time

#pin definition
motor2A = 11
motor2B = 13

#pin setting
GPIO.setmode(GPIO.BOARD)
GPIO.setup(motor2A, GPIO.OUT)
GPIO.setup(motor2B, GPIO.OUT)

#motor control
GPIO.output(motor2A, GPIO.LOW)
GPIO.output(motor2B, GPIO.LOW)
time.sleep(5)


GPIO.output(motor2A, GPIO.LOW)
GPIO.output(motor2B, GPIO.HIGH)
time.sleep(5)

#pin reset
GPIO.cleanup()
