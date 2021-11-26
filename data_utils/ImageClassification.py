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
        image_name = self.images[idx]
        # image = np.asarray(Image.open(image))/255
        image = np.asarray(Image.open(image))
        # image = image.astype(np.uint8)

        transforms = T.Compose([T.ToPILImage(), T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image = transforms(image)

        label = image_name[:image_name.find('_')]
        label = torch.tensor([self.unique_words.index(label)])
        label = F.one_hot(label, self.num_classes)

        return (image, label)

if __name__ == '__main__':
    image_dir = '../data'
    dataset = ImageClassificationDataset(image_dir)
    print(dataset.num_classes)
    for data in dataset:
        image, label_one_hot = data

        label_index = torch.argmax(label_one_hot).item()
        label = dataset.unique_words[label_index]
        print(image.shape, label_index, label)

        inv_normalize = T.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
        image = inv_normalize(image)
        image = torch.permute(image, [1,2,0])
        image = image.numpy()*255
        image = image.astype(np.uint8)
        plt.imshow(image)
        plt.title(label)
        plt.show()




