import torch
import torchvision.transforms as T
import torch.nn.functional as F
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
        self.index_to_char_mapping[28] = '<pad>'

        self.char_to_index_mapping = {v:k for k,v in self.index_to_char_mapping.items()}

        self.max_seq_length = self.get_max_seq_length(self.images)

    def __len__(self):
        return len(self.images)

    def get_max_seq_length(self, images):
        max_len = 0
        for file in images:
            name = file[:file.find('.jpg')]
            length = len(name)
            if length > max_len:
                max_len = length
        return max_len + 2 # adding 2 coz start and end tokens

    def label_tensor_to_char(self, char_sequence):
        label = [self.index_to_char_mapping[torch.argmax(tensor).item()] for tensor in char_sequence]
        return label

    def __getitem__(self, idx):
        image_name = self.images[idx]
        if type(image_name)==list:
            image_name = image_name[0]
        image = os.path.join(self.image_dir, image_name)
        image = np.asarray(Image.open(image))

        transforms = T.Compose([T.ToPILImage(), T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transforms(image)

        label = image_name[:image_name.find('_')].lower()
        label = [char for char in label]
        label.insert(0, '<start>')
        label.append('<end>')

        for i in range(self.max_seq_length - len(label)):
            label.append('<pad>')

        # one hot encode every character in the label
        label = [torch.tensor(self.char_to_index_mapping[char]) for char in label]
        label = torch.stack([F.one_hot(char_tensor, len(self.index_to_char_mapping)) for char_tensor in label])
        label = label.type(torch.FloatTensor)

        return (image, label, image_name)

if __name__ == '__main__':
    image_dir = '../data'
    dataset = ImageToSequenceDataset(image_dir)

    for data in dataset:
        image, label = data

        label = [dataset.index_to_char_mapping[torch.argmax(tensor).item()] for tensor in label]

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