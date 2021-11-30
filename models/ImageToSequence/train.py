import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ImageToSequenceModel import ImageToSequenceModel
from data_utils.ImageToSequence import ImageToSequenceDataset
from data_utils.misc_utils import norm_tensor_to_img, get_start_token_tensor

from tqdm import tqdm

torch.manual_seed(7)

def train(model, train_dataset, name=None):
    epochs = 1000
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    num_decoder_steps = train_dataset.max_seq_length

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    pbar = tqdm(train_loader)

    for epoch in range(epochs):
        for data in pbar:

            optimizer.zero_grad()
            image, label = data
            loss = 0

            image_embedding = model.image_encoder(image)
            # for first decoder timestep, feed in start token
            prev_char_sequence = get_start_token_tensor(train_dataset)

            for i in range(num_decoder_steps-1):
                flattened_char_sequence = torch.unsqueeze(torch.flatten(prev_char_sequence), dim=0)
                decoder_input = torch.cat([image_embedding, flattened_char_sequence], dim=1)
                decoder_output = model.decoder(decoder_input)
                prev_char_sequence[i+1] = decoder_output

                target = torch.unsqueeze(torch.argmax(label[0][i+1]), dim=0)
                loss+=criterion(decoder_output, target)

            loss.backward()
            optimizer.step()

            pbar.set_description('Epoch: {} Loss: {}'.format(epoch+1, loss))

if __name__ == '__main__':
    image_dir = '../../data'
    train_dataset = ImageToSequenceDataset(image_dir)

    model = ImageToSequenceModel(max_seq_length=train_dataset.max_seq_length, image_embedding_dim=512)

    train(model, train_dataset)
