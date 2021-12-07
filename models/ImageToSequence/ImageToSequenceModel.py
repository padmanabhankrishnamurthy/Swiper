import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

from data_utils.ImageToSequence import ImageToSequenceDataset
from BeamSearchDecoder import BeamDecoder
from data_utils.misc_utils import label_tensor_to_char

class ImageToSequenceModel(nn.Module):
    def __init__(self, max_seq_length, image_embedding_dim=512, device='cpu', beam=False):
        '''

        :param max_seq_length: maximum word length, i.e, maximum number of characters in a word in the training data
        :param image_height: height of all input images
        :param image_width: width of all input images
        :param image_embedding_dim: embedding dimension of image encoder
        '''

        super(ImageToSequenceModel, self).__init__()
        self.max_seq_length = max_seq_length
        self.image_embedding_dim = image_embedding_dim

        self.image_encoder = self.get_image_encoder(self.image_embedding_dim)
        self.decoder = self.get_decoder(device=device, beam=beam)

    def get_image_encoder(self, embedding_dim):
        '''
        Use MobileNetV2's feature extractors as the image encoder, set the output_size of the last layer of the feature extractor to embedding_dim
        :param height: height of input image
        :param width: width of input image
        :param embedding_dim: embedding dimension
        :return: All but final layer of MobileNetV2, where final classifier is replaced with one linear layer that projects the extracted features to an embedding_dim-d space
        NOTE : all encoder parameters but for last projection layer are frozen - can be changed based on need
        '''

        encoder = models.mobilenet_v2(pretrained=True)
        encoder.classifier = nn.Linear(in_features=1280, out_features=embedding_dim)

        # freeze all params except for last projection layer
        for param in encoder.parameters():
            param.requires_grad = False
        for param in encoder.classifier.parameters():
            param.requires_grad = True

        return encoder

    def get_decoder(self, device, beam=False):
        '''
        :return: DecoderRNN object
        '''

        if beam:
            decoder = BeamDecoder(embed_size=64, hidden_size=512, vocab_size=29, device=device)
        else:
            decoder = DecoderRNN(embed_size=64, hidden_size=512, vocab_size=29, device=device)
        return decoder

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, device='cpu'):
        super(DecoderRNN, self).__init__()

        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device

        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)

        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

        # activations
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, label=None, max_seq_len=18, teacher_forcing=False):

        # batch size
        batch_size = features.size(0)

        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(self.device)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(self.device)

        # define the output tensor placeholder
        outputs = torch.empty((batch_size, max_seq_len, self.vocab_size))

        # embed the captions
        if label:
            label = torch.stack([torch.argmax(char) for char in label[0]])
            label_embed = self.embed(label)
            label_embed = torch.unsqueeze(label_embed, dim=0)

        # decoder loop
        for t in range(max_seq_len):

            # for the first time step the input is the feature vector
            # if t==0:
            hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))

            # for the 2nd+ time step
            # else:
            #     if teacher_forcing:
            #         hidden_state, cell_state = self.lstm_cell(label_embed[:, t, :], (hidden_state, cell_state))
            #     else:
            #         output_indices = torch.stack([torch.argmax(char) for char in outputs[0]]).to(self.device)
            #         output_embed = self.embed(output_indices).to(self.device)
            #         output_embed = torch.unsqueeze(output_embed, dim=0)
            #         hidden_state, cell_state = self.lstm_cell(output_embed[:, t-1, :], (hidden_state, cell_state))

            # output of the attention mechanism
            out = self.fc_out(hidden_state)

            # build the output tensor
            outputs[:, t, :] = out

        return outputs


if __name__ == '__main__':
    dataset = ImageToSequenceDataset('about_data')
    # dataset = ImageToSequenceDataset('eval_data')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    ckpt = 'checkpoints/img2seq_no_forcing_3000_0.002223424846306443.pth'
    model = ImageToSequenceModel(max_seq_length=18, image_embedding_dim=64)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.train()

    ctr = 0

    for data in dataloader:
        
        if ctr == 10:
            break

        image, label, image_name = data
        print(image.shape)

        image_embeddings = model.image_encoder(image)
        print(image_embeddings.shape)

        result = model.decoder(image_embeddings)

        print(label_tensor_to_char(label[0]), label_tensor_to_char(result[0]))
        ctr+=1
        # break

