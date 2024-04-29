import copy
import argparse
import itertools
from collections import Counter
from collections import deque

from picamera2 import Picamera2, Preview

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier

########################################################
########################################################
########################################################

#####STEPPER SETUP START######

import time
import board
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit

kit = MotorKit(i2c=board.I2C(), address=0x60)

current_position = 0  # Starting position of the stepper motor


#####STEPPER SETUP END######

########################################################
########################################################
########################################################

#####SERVO-MOTOR SETUP START######

import RPi.GPIO as GPIO

SERVO_PIN = 21
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50) # setting frequency to 50Hz
pwm.start(0)

last_servo_angle = 90

#####SERVO-MOTOR SETUP END######

########################################################


def map_number(n, start1, stop1, start2, stop2):
    return ((n-start1)/(stop1-start1))*(stop2-start2)+start2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


########################################################
########################################################
#########START SERVO MOTOR COMPONENT################
def set_servo_angle(angle):
    duty = (angle / 90)*5 + 5.5
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)

#########END SERVO MOTOR COMPONENT##################
########################################################
########################################################


########################################################
########################################################
#########START STEPPER MOTOR COMPONENT##################

def move_stepper(target_position):
    global current_position
    global last_servo_angle

    target_position = max(0, min(target_position, 390))  # constrain the target position within the range 0 to 390

    print("target_position:", target_position)

    steps_needed = target_position - current_position

    if steps_needed > 0:
        direction = stepper.BACKWARD
        b = 1
    else:
        direction = stepper.FORWARD
        b = -1

    i = current_position
    # move the stepper motor the required number of steps
    for _ in range(abs(steps_needed)):
        kit.stepper2.onestep(direction=direction, style=stepper.DOUBLE)
        i+=b
        desired_angle = 90 if i <= 30 else 40
        if last_servo_angle != desired_angle:
            set_servo_angle(desired_angle)
            last_servo_angle = desired_angle
        time.sleep(0.01)

    # update the current position
    current_position = target_position

#########END STEPPER MOTOR COMPONENT####################
########################################################
########################################################







def main():
    try:
        # last_servo_angle = 90 # to prevent calling set_servo_angle() again and again
        # global last_servo_angle
        # set_servo_angle(last_servo_angle)
        print("inside main")
        pinch_recognized = False # flag, mainly not to have "pinch" printed infinite times
        pinch_up_detected = False
        prev_fingertip_y = float('inf') # setting initial val for fingertip's y coordinate to infinity (cuz the y grows from top to bottom)
        fingertip_y = 0

        # Argument parsing #################################################################
        args = get_args()

        # cap_device = args.device
        # cap_width = args.width
        # cap_height = args.height

        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence


        # Camera preparation ###############################################################
        camera = Picamera2()
        capture_config = camera.create_still_configuration()
        camera.configure(capture_config)
        camera.start()
        print("aftter camera")


        # Model load #############################################################
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()


        # FPS Measurement ########################################################
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        loop_counter = 0

        while True:
            # print("inside true")



            image = camera.capture_array()

            fps = cvFpsCalc.get()

            # Camera capture #####################################################

            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Detection implementation #############################################################
            # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):

                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)

                    # print("landmarks:", pre_processed_landmark_list)

                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 0 and not pinch_recognized:
                        print("\n")
                        print("Pinch Gesture Recognized!")
                        print("\n")
                        pinch_recognized = True


                    elif hand_sign_id != 0:
                        pinch_recognized = False

                    if hand_sign_id == 0 and pinch_recognized:
                        fingertip_y = landmark_list[8][1]

                        if loop_counter > 1 and abs(fingertip_y - prev_fingertip_y) <= 100:
                            continue

                        # below, <=100 is needed to ignore rapid movement
                        if abs(fingertip_y - prev_fingertip_y) >= 50: # if prev_fingertip_y is None or ... <- might be good just for additional check
                            # print("fingertip-y:", fingertip_y)

                            loop_counter = 1

                            mapped_value = map_number(fingertip_y, 1700, 300, 0, 390)
                            mapped_value = round(mapped_value) # no float
                            move_stepper(mapped_value)
                            # print("mapped_val", mapped_value)
                            prev_fingertip_y = fingertip_y







    finally:
        move_stepper(0)
        kit.stepper2.release()
        print("Stepper motor released.")
        set_servo_angle(90)
        pwm.stop()
        GPIO.cleanup()  # Clean up GPIO to ensure all pins are reset properly
        print("GPIO cleaned up.")





def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list




if __name__ == '__main__':
    main()

