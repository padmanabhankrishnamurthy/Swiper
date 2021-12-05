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

def get_words(words_file):
    unique_words = [word.strip() for word in open(words_file, 'r')]
    return unique_words

def get_start_sequence_tensor(train_dataset):
    start_token = ['<pad>'] * train_dataset.max_seq_length
    start_token[0] = '<start>'
    start_token = [torch.tensor(train_dataset.char_to_index_mapping[char]) for char in start_token]
    start_token = torch.stack([F.one_hot(char_tensor, len(train_dataset.index_to_char_mapping)) for char_tensor in start_token])
    start_token = start_token.type(torch.FloatTensor)
    return start_token

def get_index_to_char_mapping():
    index_to_char_mapping = {0: '<start>'}
    index_to_char_mapping.update({i + 1: chr(c) for i, c in enumerate(range(ord('a'), ord('z') + 1))})
    index_to_char_mapping[27] = '<end>'
    index_to_char_mapping[28] = '<pad>'
    return index_to_char_mapping

def label_tensor_to_char(char_sequence):
    index_to_char_mapping = get_index_to_char_mapping()
    label = [index_to_char_mapping[torch.argmax(tensor).item()] for tensor in char_sequence]
    return label

def clean_decoder_output(decoder_output):
    if '<end>' in decoder_output:
        decoder_output = decoder_output[decoder_output.index('<start>') + 1:decoder_output.index('<end>')]
    else:
        decoder_output = decoder_output[decoder_output.index('<start>') + 1:decoder_output.index('<pad>')]
    decoder_output = ''.join(decoder_output)

    return decoder_output