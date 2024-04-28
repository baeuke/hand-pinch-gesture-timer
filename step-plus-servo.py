import time
import board
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit

import RPi.GPIO as GPIO
import time

kit = MotorKit(i2c=board.I2C(), address=0x60)

SERVO_PIN = 21
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

