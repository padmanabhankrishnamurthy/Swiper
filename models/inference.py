import os

import torch
from torch.utils.data import DataLoader

from data_utils.ImageClassification import ImageClassificationDataset
import numpy as np
import torchvision.transforms as T
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def load_model(words_list='/Users/padmanabhankrishnamurthy/PycharmProjects/Swiper/words.txt', checkpoint='/Users/padmanabhankrishnamurthy/PycharmProjects/Swiper/stored_models/classifier_1000.pth'):
    words = [word.strip() for word in open(words_list, 'r')]
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=len(words), bias=True)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    return model, words

if __name__ == '__main__':
    data_dir = '../eval_data'
    words_list = '../words.txt'
    val_set = ImageClassificationDataset(data_dir, words_list)
    val_loader = DataLoader(val_set, batch_size=1)

    model, words = load_model()

    correct = 0
    for index, data in enumerate(val_loader):
        image, true_label = data

        with torch.no_grad():
            model.eval()
            output = model(image)
            label_index = torch.argmax(output)
            label = val_set.unique_words[label_index]
            true_label = val_set.unique_words[torch.argmax(true_label)]
            print(index, label, true_label)
            if label == true_label:
                correct+=1

    print('Accuracy : {}'.format(correct/len(val_loader)))


