import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import sys

sys.path.insert(0, '/content/Swiper')

from ImageToSequenceModel import ImageToSequenceModel
from data_utils.ImageToSequence import ImageToSequenceDataset
from data_utils.misc_utils import norm_tensor_to_img, get_start_sequence_tensor

from tqdm import tqdm
import numpy as np

torch.manual_seed(7)


def train(model, train_dataset, save_path=None, save_name=None):
    epochs = 1000
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    num_decoder_steps = train_dataset.max_seq_length

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader)

        for data in pbar:
            optimizer.zero_grad()
            image, label, image_name = data

            model.to('cuda')
            image_embedding = model.image_encoder(image.to('cuda'))
            decoder_output = model.decoder(image_embedding.to('cuda'), label.to('cuda'))
            target = torch.stack([torch.argmax(char) for char in label[0]])
            loss = criterion(decoder_output[0], target)

            epoch_loss += loss

            loss.backward()
            optimizer.step()
            pbar.set_description('Epoch: {} Loss: {}'.format(epoch + 1, loss))

        if (epoch + 1) % 10 == 0:
            print('Epoch: {} | Loss: {}'.format(epoch + 2001, epoch_loss / len(train_dataset)))
            if save_path and save_name:
                torch.save(model.state_dict(), '{}/{}_{}_{}.pth'.format(save_path, save_name, epoch + 2001,
                                                                        epoch_loss / len(train_dataset)))


if __name__ == '__main__':
    image_dir = '/content/drive/MyDrive/data'
    train_dataset = ImageToSequenceDataset(image_dir)

    model = ImageToSequenceModel(max_seq_length=train_dataset.max_seq_length, image_embedding_dim=64)
    ckpt = '/content/drive/MyDrive/swiper_models/img2seq_no_forcing/img2seq_no_forcing_2000_0.3214365839958191.pth'
    model.load_state_dict(torch.load(ckpt))

    save_path = '/content/drive/MyDrive/swiper_models/img2seq_no_forcing'
    save_name = 'img2seq_no_forcing'
    train(model, train_dataset, save_name, save_path)
