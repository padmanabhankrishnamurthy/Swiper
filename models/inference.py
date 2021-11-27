import torch

from data_utils.ImageClassification import ImageClassificationDataset
import numpy as np
import torchvision.transforms as T
import torch.nn as nn
import torchvision.models as models
from PIL import Image


data_dir = '../data'
words_list = '../words.txt'
train_set = ImageClassificationDataset(data_dir, words_list)
image = '../data/about_3.jpg'
checkpoint = '../stored_models/classifier_300.pth'

image = np.asarray(Image.open(image))
transforms = T.Compose([T.ToPILImage(), T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

image = transforms(image)

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(in_features=1280, out_features=train_set.num_classes, bias=True)

# state_dict = torch.load(checkpoint, map_location='cpu')
# print(state_dict)

model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

with torch.no_grad():
    model.eval()
    image = torch.unsqueeze(image, dim=0)
    output = model(image)
    label_index = torch.argmax(output)
    label = train_set.unique_words[label_index]
    print(label_index, label)

print(train_set.unique_words)

