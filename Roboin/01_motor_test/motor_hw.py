#library
import RPi.GPIO as GPIO
import time

#pin definition
motor1A = 16
motor1B = 18
motor2A = 13
motor2B = 15

#pin setting
GPIO.setmode(GPIO.BOARD)
GPIO.setup(motor1A, GPIO.OUT)
GPIO.setup(motor1B, GPIO.OUT)
GPIO.setup(motor2A, GPIO.OUT)
GPIO.setup(motor2B, GPIO.OUT)

#motor control
GPIO.output(motor1A, GPIO.HIGH)
GPIO.output(motor1B, GPIO.LOW)
GPIO.output(motor2A, GPIO.HIGH)
GPIO.output(motor2B, GPIO.LOW)
time.sleep(10)


#GPIO.output(motor1A, GPIO.LOW)
#GPIO.output(motor1B, GPIO.HIGH)
#GPIO.output(motor2A, GPIO.LOW)

#GPIO.output(motor2B, GPIO.HIGH)
#time.sleep(5)

#pin reset
GPIO.cleanup()
