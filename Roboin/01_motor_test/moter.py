#library
import RPi.GPIO as GPIO
import time

#pin definition
motor1A = 16
motor1B = 18

#pin setting
GPIO.setmode(GPIO.BOARD)
GPIO.setup(motor1A, GPIO.OUT)
GPIO.setup(motor1B, GPIO.OUT)

#motor control
GPIO.output(motor1A, GPIO.LOW)
GPIO.output(motor1B, GPIO.LOW)
time.sleep(5)


GPIO.output(motor1A, GPIO.LOW)
GPIO.output(motor1B, GPIO.HIGH)
time.sleep(2)

#pin reset
GPIO.cleanup()
