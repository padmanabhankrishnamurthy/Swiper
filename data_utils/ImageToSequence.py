import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageToSequenceDataset(Dataset):
    def __init__(self, image_dir):
        super(ImageToSequenceDataset, self).__init__()
        self.image_dir = image_dir
        self.images = [file for file in os.listdir(self.image_dir) if '.jpg' in file]

        self.index_to_char_mapping = {0:'<start>'}
        self.index_to_char_mapping.update({i+1:chr(c) for i,c in enumerate(range(ord('a'),ord('z')+1))})
        self.index_to_char_mapping[27] = '<end>'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = os.path.join(self.image_dir, self.images[idx])
        image_name = self.images[idx]
        image = np.asarray(Image.open(image))

        transforms = T.Compose([T.ToPILImage(), T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transforms(image)

        label = image_name[:image_name.find('_')].lower()
        label = [char for char in label]
        label.insert(0, '<start>')
        label.append('<end>')

        return (image, label)

if __name__ == '__main__':
    image_dir = '../data'
    dataset = ImageToSequenceDataset(image_dir)

    for data in dataset:
        image, label = data

        print(image.shape, label)

        inv_normalize = T.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
        image = inv_normalize(image)
        image = torch.permute(image, [1, 2, 0])
        image = image.numpy() * 255
        image = image.astype(np.uint8)
        plt.imshow(image)
        plt.title(label)
        plt.show()
        break