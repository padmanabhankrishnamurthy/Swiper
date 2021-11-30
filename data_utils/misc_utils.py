import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def norm_tensor_to_img(image, label=None):
    inv_normalize = T.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
    image = inv_normalize(image)
    image = torch.permute(image, [1, 2, 0])
    image = image.numpy() * 255
    image = image.astype(np.uint8)
    plt.imshow(image)
    if label:
        plt.title(label)
    plt.show()

def get_start_token_tensor(train_dataset):
    start_token = ['<pad>'] * train_dataset.max_seq_length
    start_token[0] = '<start>'
    start_token = [torch.tensor(train_dataset.char_to_index_mapping[char]) for char in start_token]
    start_token = torch.stack([F.one_hot(char_tensor, len(train_dataset.index_to_char_mapping)) for char_tensor in start_token])
    start_token = start_token.type(torch.FloatTensor)
    return start_token