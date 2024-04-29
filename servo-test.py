import RPi.GPIO as GPIO
import time

SERVO_PIN = 21
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50) # setting frequency to 50Hz
pwm.start(0)


def set_angle(angle):
    duty = (angle / 90)*5 + 5.5
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)


try:
    while True:
        angle_input = input("Enter angle (0 to 90 degrees, 'q' to quit): ")
        if angle_input.lower() == 'q':
            break
        angle = int(angle_input)
        if angle < 0 or angle > 90:
            print("Angle must be between 0 and 90 degrees.")
            continue
        set_angle(angle)

except KeyboardInterrupt:
    pass

finally:
    pwm.stop()
    GPIO.cleanup()




