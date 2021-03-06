import cv2
import mediapipe as mp
import numpy as np
from colour import Color

import torch
import torchvision.models as models
import torch.nn as nn

from models.ImageToSequence.ImageToSequenceModel import ImageToSequenceModel
import models.ImageToSequence.inference as Im2Seq
import models.ImageClassification.inference as Im2Cls

torch.manual_seed(7)
np.random.seed(7)


GESTURE_WRITE = 1
GESTURE_STOP = 2
GESTURE_DELETE = 3

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

def load_classification_model(words_list='words.txt', checkpoint='checkpoints/classifier_1000.pth'):
    words = [word.strip() for word in open(words_list, 'r')]
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=len(words), bias=True)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    return model, words

def load_sequence_model(checkpoint='checkpoints/img2seq_only_image_decoded_4500_0.0002415632625343278.pth'):
    model = ImageToSequenceModel(max_seq_length=18, image_embedding_dim=64)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    return model

def gen_trail(index_finger_tip_points, colours, image_shape, crop_shape):
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
    return trail_image

def distance(a,b):
    return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y)

def gesture_control(landmarks):
    gesture = None
    landmarks = landmarks[0]
    pf = distance(landmarks.landmark[6], landmarks.landmark[0]) < distance(landmarks.landmark[8], landmarks.landmark[0])
    c_f3 = all(distance(landmarks.landmark[l+2], landmarks.landmark[0]) < distance(landmarks.landmark[l], landmarks.landmark[0]) for l in [10,14,18])
    swipe_left = all(landmarks.landmark[4].x < landmarks.landmark[l].x for l in [8,12,16,20])

    if pf and not c_f3:
        if swipe_left:
            gesture = GESTURE_DELETE
        else:
            gesture = GESTURE_STOP
    elif pf and c_f3:
        gesture = GESTURE_WRITE

    return gesture

def detect_hands(model_type):
    index_finger_tip_points = []
    display_text = ""
    curr_ges = GESTURE_DELETE
    prev_ges = curr_ges

    cap = cv2.VideoCapture(0)

    if model_type == 'Classification':
        SwypNET, word_list = load_classification_model()
    elif model_type == 'Sequence':
        SwypNET = load_sequence_model()
    SwypNET.eval()

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

        # overlay keyboard on the center of the frame - bottom corners seem to be screwing up hand detection
        # this needs to be done before trail generation, otherwise keyboard obscures finger and trail
        top_left_x = int(h/4 - keyboard_height/4)
        top_left_y = int(w/2 - keyboard_width/2)
        crop_shape = [[top_left_x, top_left_x+keyboard_height],[top_left_y,top_left_y+keyboard_width]]
        image[top_left_x:top_left_x+keyboard_height, top_left_y:top_left_y+keyboard_width] = keyboard

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks)==1:
            curr_ges = gesture_control(results.multi_hand_landmarks)
            index_finger_tip = results.multi_hand_landmarks[0].landmark[8]
            index_finger_tip = (int(index_finger_tip.x*w), int(index_finger_tip.y*h))
            cv2.circle(image, index_finger_tip, 5, (255, 0, 255), cv2.FILLED)

            # capture index finger trail
            if curr_ges == GESTURE_WRITE and len(results.multi_hand_landmarks)==1:
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

            elif prev_ges == GESTURE_WRITE and curr_ges == GESTURE_STOP:
                trail_image = gen_trail(index_finger_tip_points, colours, image.shape, crop_shape)
                trail_image = 255 * trail_image
                trail_image = trail_image.astype(np.uint8)
                trail_image = np.asarray(trail_image)
                if model_type == 'Classification':
                    word = Im2Cls.infer(trail_image, SwypNET, words = word_list, transform=True)
                    display_text+=word
                elif model_type == 'Sequence':
                    temp_text = Im2Seq.infer(trail_image, SwypNET, transform=True)
                    temp_text = [x for x in temp_text if '<' not in x]
                    display_text += "".join(temp_text)
                display_text += " "
                # reset trail for next word if palm raised
                index_finger_tip_points = []

            elif curr_ges == GESTURE_DELETE:
                display_text = ""
        
        prev_ges = curr_ges
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        #crop image to only display keyboard and bottom console area
        image = image[top_left_x:top_left_x+keyboard_height+50, top_left_y:top_left_y+keyboard_width]
        image[keyboard_height:, :] = np.zeros((50, keyboard_width, 3))

        # display text if any - used to display words during data collection
        cv2.putText(image, display_text, org=(10, image.shape[0] - 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, color=[0,0,255])

        cv2.imshow('Swiper', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()

if __name__ == '__main__':
    model_type = 'Classification'
    # model_type = 'Sequence'
    detect_hands(model_type=model_type)
