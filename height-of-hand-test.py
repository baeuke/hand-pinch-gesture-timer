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

print("before main")








def main():
    print("inside main")
    pinch_recognized = False
    pinch_up_detected = False
    prev_fingertip_y = float('inf')
    fingertip_y = 0

    args = get_args()

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    camera = Picamera2()
    capture_config = camera.create_still_configuration()
    camera.configure(capture_config)
    camera.start()
    print("after camera")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        print("inside true")
        image = camera.capture_array()
        fps = cvFpsCalc.get()

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

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
                    if abs(fingertip_y - prev_fingertip_y) >= 50:
                        mapped_value = map_number(fingertip_y, 600, 1300, 3, 12)
                        # Assuming you have defined 'pwm' somewhere
                        # pwm.ChangeDutyCycle(mapped_value)
                        print("fingertip-y:", fingertip_y)
                        prev_fingertip_y = fingertip_y

        # Display the camera feed
        cv.imshow('Camera Feed', image)
        cv.waitKey(1)  # Wait for a short time to allow OpenCV to handle events

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

if __name__ == '__main__':
    main()
