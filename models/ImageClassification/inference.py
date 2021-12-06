from models.ImageClassification.ImageClassificationModel import ImageClassificationDataset
from data_utils.ImageClassification import ImageClassificationDataset
from data_utils.misc_utils import get_words

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models

import os
import numpy as np
from PIL import Image

def load_model(num_classes, checkpoint):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    return model

def infer(image, model, words=None, words_file=None, transform = False):

    # transform image if image is not a torch tensor
    if isinstance(image, str):
        image = np.asarray(Image.open(image))
    
    if transform:
        transforms = T.Compose([T.ToPILImage(), T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transforms(image)
        image = torch.unsqueeze(image, dim=0)

    if words_file:
        words = get_words(words_file)

    model.eval()
    output = model(image)
    label = words[torch.argmax(output)]
    return label


def evaluate(eval_set, model):
    # param eval_set is a torch dataset

    dataloader = DataLoader(eval_set, batch_size=1, shuffle=False)
    correct = 0

    for data in dataloader:
        image, true_label, image_name = data
        predicted = infer(image, model, words=eval_set.unique_words)

        # true_label = image_name[0].split('_')[0]
        true_label = eval_set.unique_words[torch.argmax(true_label)]
        if predicted == true_label:
            correct += 1

        print(true_label, predicted, predicted==true_label, '\n')

    print('Accuracy : {}'.format(correct/len(dataloader)))

if __name__ == '__main__':
    image_dir = '../../eval_data'
    words_list = '../../words.txt'
    ckpt = '../../checkpoints/classifier_1000.pth'
    eval_set = ImageClassificationDataset(image_dir, words_list)
    words = get_words(words_list)

    model = load_model(num_classes=len(words), checkpoint=ckpt)

    # run evaluation
    # evaluate(eval_set, model)

    # run single image inference
    result = infer(os.path.join(image_dir, 'hey_2.jpg'), model, words_file=words_list)
    print(result)