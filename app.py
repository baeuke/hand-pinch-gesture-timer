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

last_servo_angle = 75
#####SERVO-MOTOR SETUP END######


########################################################
########################################################
########################################################


#####RANGE SETUP START######
# start is the vertical bottom
# end is the vertical top

bottle_range_start = 0
bottle_range_end = 390

# hand range is basically camera's coordinate range which increases from top to bottom
hand_range_start = 1700
hand_range_end = 300
#####RANGE SETUP END######


########################################################
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
    duty = (angle / 90)*5 + 5.5 # this will give the [5.5, 10.5] range for 0 to 90 degrees
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

    target_position = max(bottle_range_start, min(target_position, bottle_range_end))  # constrain the target position within the range 0 to 390

    print("target_position:", target_position)

    steps_needed = target_position - current_position

    if steps_needed > 0:
        direction = stepper.BACKWARD
        b = 1 # positive incrementer for the 0 -> 390 range
    else:
        direction = stepper.FORWARD
        b = -1 # negative incrementer for the 0 -> 390 range

    i = current_position
    # move the stepper motor the required number of steps
    for _ in range(abs(steps_needed)):
        kit.stepper2.onestep(direction=direction, style=stepper.DOUBLE)
        i+=b # incrementing 'virtual' current position to know if we need to rotate the magnet (servo)
        desired_angle = 80 if i <= 30 else 40 # so, below position of 30 we want the magnet to always be 80
        if last_servo_angle != desired_angle: #otherwise set to 40 degrees
            set_servo_angle(desired_angle)
            last_servo_angle = desired_angle
        time.sleep(0.005)

    # update the current position
    current_position = target_position
#########END STEPPER MOTOR COMPONENT####################
########################################################
########################################################


########################################################
########################################################
#########START TIMED STEPPER COMPONENT##################
# this component is for using stepper after timer was set
def timed_stepper(target_position):
    global current_position
    global last_servo_angle

    target_position = max(bottle_range_start, min(target_position, bottle_range_end))  # constrain the target position within the range 0 to 390

    print("target_position:", target_position)

    steps_needed = target_position - current_position

    if steps_needed > 0:
        direction = stepper.BACKWARD
    else:
        direction = stepper.FORWARD


    # move the stepper motor the required number of steps
    for _ in range(abs(steps_needed)):
        kit.stepper2.onestep(direction=direction, style=stepper.DOUBLE)
        time.sleep(0.005)

    # update the current position
    current_position = target_position
#########END TIMED STEPPER COMPONENT####################
########################################################
########################################################


########################################################
########################################################
#########START TIMER COMPONENT##########################
def set_timer(minutes):
    seconds = minutes * 60
    new_servo_angle = 60
    while seconds:
        mins, secs = divmod(seconds, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        seconds -= 1
        if new_servo_angle == 60:
            set_servo_angle(60)
            new_servo_angle = 61
        else:
            set_servo_angle(61)
            new_servo_angle = 60

        kit.stepper2.onestep(direction=stepper.FORWARD, style=stepper.DOUBLE)

    print("Time's up!")
#########END TIMER COMPONENT####################
########################################################
########################################################

def main():
    try:
        open_palm_flag = 0 # to break from the main loop
        rapid_range_flag = 0 # to continue in the main loop
        print("inside main")
        pinch_recognized = False # flag, mainly not to have "pinch" printed infinite times
        pinch_up_detected = False
        prev_fingertip_y = float('inf') # setting initial val for fingertip's y coordinate to infinity (cuz the y grows from top to bottom)
        fingertip_y = 0

        # Argument parsing #################################################################
        args = get_args()

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

        loop_counter = 0 # for knowing the first iteration of the loop

        while True:
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

                    if hand_sign_id == 2 and current_position > 50:
                        print("OPEN PALM")
                        time.sleep(2)
                        open_palm_flag = 1
                        break


                    elif hand_sign_id != 0:
                        pinch_recognized = False

                    if hand_sign_id == 0 and pinch_recognized:
                        fingertip_y = landmark_list[8][1]

                        if loop_counter > 1 and abs(fingertip_y - prev_fingertip_y) >= 100:
                            rapid_range_flag = 1
                            break

                        # below, <=100 is needed to ignore rapid movement
                        if abs(fingertip_y - prev_fingertip_y) >= 50: # if prev_fingertip_y is None or ... <- might be good just for additional check
                            # print("fingertip-y:", fingertip_y)
                            loop_counter = 1

                            mapped_value = map_number(fingertip_y, hand_range_start, hand_range_end, bottle_range_start, bottle_range_end)
                            mapped_value = round(mapped_value) # no float
                            move_stepper(mapped_value)
                            # print("mapped_val", mapped_value)
                            prev_fingertip_y = fingertip_y

                if open_palm_flag == 1:
                    break



        print ("breeaked")
        final_position = current_position
        print("final position:", final_position)
        move_stepper(0)
        # 10 minutes:
        if 255 < final_position <= 305:
            set_servo_angle(80) #bigger blob
            timed_stepper(280)
            set_timer(10)
        # 20 minutes:
        elif 305 < final_position <= 325:
            set_servo_angle(80) #bigger blob
            timed_stepper(320)
            set_timer(20)
        # 30 minutes:
        elif 325 < final_position <= 390:
            set_servo_angle(80) #bigger blob
            timed_stepper(375)
            set_timer(30)
        # 2 minutes:
        elif 70 <= final_position <= 110:
            set_servo_angle(60) #smaller blob
            timed_stepper(105)
            set_timer(2)
        # 3 minutes:
        elif 110 < final_position <= 185:
            set_servo_angle(60) #smaller blob
            timed_stepper(175)
            set_timer(3)
        # 5 minutes:
        elif 185 < final_position <= 255:
            set_servo_angle(60) #smaller blob
            timed_stepper(215)
            set_timer(5)


    finally:
        move_stepper(0)
        kit.stepper2.release()
        print("Stepper motor released.")
        set_servo_angle(75)
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

