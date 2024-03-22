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
    pinch_recognized = False # flag, mainly not to have "pinch" printed infinite times
    pinch_up_detected = False
    prev_fingertip_y = float('inf') # setting initial val for fingertip's y coordinate to infinity (cuz the y grows from top to bottom)
    fingertip_y = 0

    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence


    # Camera preparation ###############################################################
    camera = Picamera2()
    capture_config = camera.create_still_configuration()
    camera.configure(capture_config)
    print("aftter camera")
    # camera.resolution = (960, 540)
    # camera.start_preview(Preview.NULL)
    # camera.resolution = (cap_width, cap_height)
    # camera.framerate = 24
    # rawCapture = PiRGBArray(camera, size=(cap_width, cap_height))

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

    mode = 0


    while True:
        print("inside true")
        image = camera.capture_array()
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

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

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 0 and not pinch_recognized:
                    print("Pinch Gesture Recognized!")
                    pinch_recognized = True


                elif hand_sign_id != 0:
                    pinch_recognized = False

                if hand_sign_id == 0 and pinch_recognized:
                    fingertip_y = landmark_list[8][1]
                    if abs(fingertip_y - prev_fingertip_y) >= 50: # if prev_fingertip_y is None or ... <- might be good just for additional check
                        print("fingertip-y:", fingertip_y)
                        prev_fingertip_y = fingertip_y


        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)
        rawCapture.truncate(0)

    # cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode



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
