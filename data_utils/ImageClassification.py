import torch

from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms as T

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageClassificationDataset(Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.images = os.listdir(self.images_dir)
        self.unique_words = self.get_unique_words()
        self.num_classes = len(self.unique_words)

    def get_unique_words(self):
        unique_words = set([file[:file.find('_')] for file in self.images])
        return list(unique_words)

    def __len__(self):
        return len(os.listdir(self.images_dir))

    def __getitem__(self, idx):
        image = os.path.join(self.images_dir, self.images[idx])
        image_name = image
        image = np.asarray(Image.open(image))/255

        transforms = T.Compose(T.ToPILImage(), T.Resize((256,256)), T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        image = transforms(image)

        label = image_name[:image_name.find('_')]
        label = torch.tensor([self.unique_words.index(label)])
        label = F.one_hot(label, self.num_classes)

        return (image, label)



