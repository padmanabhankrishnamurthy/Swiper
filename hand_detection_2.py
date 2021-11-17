import cv2
import mediapipe as mp
from collections import deque
import numpy as np

# media pipe objects
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def detect_hands():
    index_finger_tip_points = []

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        h, w, c = image.shape

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:

          index_finger_tip = results.multi_hand_landmarks[0].landmark[8]
          index_finger_tip = (int(index_finger_tip.x*w), int(index_finger_tip.y*h))
          cv2.circle(image, index_finger_tip, 5, (255, 0, 255), cv2.FILLED)
          index_finger_tip_points.insert(0, index_finger_tip)

          for i in range(1, len(index_finger_tip_points)):

              if index_finger_tip_points[i - 1] is None or index_finger_tip_points[i] is None:
                  continue

              cv2.line(image, index_finger_tip_points[i - 1], index_finger_tip_points[i], (0, 0, 255), thickness=5)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()

detect_hands()
