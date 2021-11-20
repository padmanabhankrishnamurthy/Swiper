import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from colour import Color

# media pipe objects
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# keyboard overlay
keyboard = cv2.imread('keyboard.png')
scale_percent = 70
keyboard_width = int(keyboard.shape[1] * scale_percent / 100)
keyboard_height = int(keyboard.shape[0] * scale_percent / 100)
# resize keyboard
keyboard = cv2.resize(keyboard, (keyboard_width, keyboard_height), interpolation=cv2.INTER_AREA)
# flip the keyboard because all video frames are ultimately flipped again before being rendered
keyboard = cv2.flip(keyboard, 1)


def palm_open(landmarks):
    # param landmarks should be a results.multi_hand_landmarks WITH NO FURTHER INDEXING

    # use middle finger open as predictor of palm open
    # see https://gist.github.com/TheJLifeX/74958cc59db477a91837244ff598ef4a for more

    landmarks = landmarks[0]
    middle_open = landmarks.landmark[12].y < landmarks.landmark[10].y and landmarks.landmark[11].y < landmarks.landmark[10].y

    return middle_open

def save_trail(index_finger_tip_points, colours, image_shape, crop_shape, name=None, path=None):
    h,w,c = image_shape
    trail_image = np.zeros((h, w, c))

    # index finger tip points also contains an erroneous stroke depicting palm opening - we need to cut out that stroke before saving
    # the slice start index, 5, has been chosen arbitrarily
    index_finger_tip_points = index_finger_tip_points[:-5]

    for i in range(1, len(index_finger_tip_points)):
        if index_finger_tip_points[i - 1] is None or index_finger_tip_points[i] is None:
            continue

        # draw trail
        colour = (colours[i].red, colours[i].green, colours[i].blue)
        cv2.line(trail_image, index_finger_tip_points[i - 1], index_finger_tip_points[i], colour, thickness=5)

    trail_image = cv2.flip(trail_image, 1)
    trail_image = trail_image[crop_shape[0][0]:crop_shape[0][1], crop_shape[1][0]:crop_shape[1][1]]
    plt.imshow(trail_image)
    plt.show()

    # save image
    if name:
        print(name)
        files = [file[:file.find('_')] for file in os.listdir(path)]
        file_dict = {file:files.count(file) for file in files}
        file_set = set(files)
        if name not in file_set:
            filename = name+'_1.jpg'
        else:
            filename = '{}_{}.jpg'.format(name, file_dict[name]+1)
        trail_image = trail_image * 255
        trail_image = trail_image.astype('uint8')
        cv2.imwrite(os.path.join(path, filename), trail_image)

def nav_gestures(landmarks):
    gesture = None
    landmarks = landmarks[0]
    opalm_up = all(landmarks.landmark[l+3].y < landmarks.landmark[l].y for l in [5,9,13,17])
    opalm_down = all(landmarks.landmark[l+3].y > landmarks.landmark[l].y for l in [5,9,13,17])
    swipe_left = all(landmarks.landmark[4].x < landmarks.landmark[l].x for l in [8,12,16,20])
    gesture = "open_palm_up: " + str(opalm_up) + " opalm_down: " + str(opalm_down) + " swipe_left: " + str(swipe_left)
    return gesture


def detect_hands(word_list=None, save_path=None):
    index_finger_tip_points = []
    word_list_index = 0
    samples_captured = 0
    samples_per_word = 5

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        display_text = word_list[word_list_index] if word_list else None
        success, image = cap.read()
        h, w, c = image.shape

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hand_detector.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # don't capture trail if 2 hands are displayed - necessary for moving from end point of word to start point of next word
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
            index_finger_tip_points = []

        # overlay keyboard on the center of the frame - bottom corners seem to be screwing up hand detection
        # this needs to be done before trail generation, otherwise keyboard obscures finger and trail
        top_left_x = int(h/4 - keyboard_height/4)
        top_left_y = int(w/2 - keyboard_width/2)
        crop_shape = [[top_left_x, top_left_x+keyboard_height],[top_left_y,top_left_y+keyboard_width]]
        image[top_left_x:top_left_x+keyboard_height, top_left_y:top_left_y+keyboard_width] = keyboard

        gesture_test = ""

        # capture index finger trail
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks)==1:

            gesture_test = nav_gestures(results.multi_hand_landmarks)

            index_finger_tip = results.multi_hand_landmarks[0].landmark[8]
            index_finger_tip = (int(index_finger_tip.x*w), int(index_finger_tip.y*h))
            cv2.circle(image, index_finger_tip, 5, (255, 0, 255), cv2.FILLED)
            index_finger_tip_points.append(index_finger_tip)

            # initialise colours for trail
            start_colour = Color('orange')
            colours = list(start_colour.range_to(Color("blue"), len(index_finger_tip_points)))

            # draw trail
            for i in range(1, len(index_finger_tip_points)):
                if index_finger_tip_points[i - 1] is None or index_finger_tip_points[i] is None:
                    continue

                colour = (colours[i].red*255, colours[i].green*255, colours[i].blue*255)
                cv2.line(image, index_finger_tip_points[i - 1], index_finger_tip_points[i], colour, thickness=5)

            # check palm open and terminate trail generation if so
            if False and palm_open(results.multi_hand_landmarks):
                # create trail on blank image and save
                if save_path:
                    save_trail(index_finger_tip_points, colours, image.shape, crop_shape, display_text, save_path)
                    samples_captured+=1
                    if samples_captured == samples_per_word:
                        word_list_index+=1 # display next word
                        samples_captured = 0 # set count of next word to 0

                # visualise trail plot without saving coz no save path provided
                else:
                    save_trail(index_finger_tip_points, colours, image.shape, crop_shape)

                # pause so that save_trail isn't called multiple times
                time.sleep(1)
                # reset trail for next word if palm raised
                index_finger_tip_points = []

        elif results.multi_hand_landmarks and len(results.multi_hand_landmarks)!=1:
            print(len(results.multi_hand_landmarks))

        else:
            print(results.multi_hand_landmarks)

        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        #crop image to only display keyboard and bottom console area
        image = image[top_left_x:top_left_x+keyboard_height+50, top_left_y:top_left_y+keyboard_width]

        # display text if any - used to display words during data collection
        if word_list:
            cv2.putText(image, display_text, org=(10, image.shape[0] - 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, color=[0,0,255])

        image[-50:,:] = 0
        cv2.putText(image, gesture_test, org=(10, image.shape[0] - 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, thickness=1, color=[255,255,0])

        cv2.imshow('Swiper', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()

if __name__ == '__main__':
    detect_hands()
