import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time

# media pipe objects
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def palm_open(landmarks):
    # param landmarks should be a results.multi_hand_landmarks WITH NO FURTHER INDEXING

    # use middle finger open as predictor of palm open
    # see https://gist.github.com/TheJLifeX/74958cc59db477a91837244ff598ef4a for more

    landmarks = landmarks[0]
    middle_open = landmarks.landmark[12].y < landmarks.landmark[10].y and landmarks.landmark[11].y < landmarks.landmark[10].y

    return middle_open

def save_trail(index_finger_tip_points, image_shape):
    print('called')
    h,w,c = image_shape
    trail_image = np.zeros((h, w, c))

    for i in range(1, len(index_finger_tip_points)):
        if index_finger_tip_points[i - 1] is None or index_finger_tip_points[i] is None:
            continue

        # draw trail
        cv2.line(trail_image, index_finger_tip_points[i - 1], index_finger_tip_points[i], (0, 0, 255), thickness=5)

    trail_image = cv2.flip(trail_image, 1)
    plt.imshow(trail_image)
    plt.show()

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

                # check palm open and terminate trail generation if so
                if palm_open(results.multi_hand_landmarks):
                    # create trail on blank image and save
                    save_trail(index_finger_tip_points, image.shape)

                    # pause so that save_trail isn't called multiple times
                    time.sleep(1)

                    index_finger_tip_points = []
                    break

                # draw trail
                cv2.line(image, index_finger_tip_points[i - 1], index_finger_tip_points[i], (0, 0, 255), thickness=5)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()

if __name__ == '__main__':
    detect_hands()
