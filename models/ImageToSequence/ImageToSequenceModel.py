import torch
import torch.nn as nn
import torchvision.models as models

class ImageToSequenceModel(nn.Module):
    def __init__(self, max_seq_length, image_embedding_dim=512):
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
        self.decoder = self.get_decoder()

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

    def get_decoder(self):
        '''
        :return: DecoderRNN object 
        '''

        decoder = DecoderRNN(embed_size=64, hidden_size=512, vocab_size=29)
        return decoder

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()

        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)

        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

        # activations
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, label):

        # batch size
        batch_size = features.size(0)

        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size))
        cell_state = torch.zeros((batch_size, self.hidden_size))

        # define the output tensor placeholder
        # print('captions ', captions.shape)
        outputs = torch.empty((batch_size, label.size(1), self.vocab_size))
        # print(outputs.shape)

        # embed the captions
        # print(captions.shape)
        label = torch.stack([torch.argmax(char) for char in label[0]])
        label_embed = self.embed(label)
        label_embed = torch.unsqueeze(label_embed, dim=0)
        # print(captions.shape, label_embed.shape)

        # pass the caption word by word
        # print(label.shape, label.size(), label.size(0), label_embed.shape)
        for t in range(label.size(0)):

            # for the first time step the input is the feature vector
            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))

            # for the 2nd+ time step, using teacher forcer
            else:
                hidden_state, cell_state = self.lstm_cell(label_embed[:, t, :], (hidden_state, cell_state))

            # output of the attention mechanism
            out = self.fc_out(hidden_state)

            # build the output tensor
            outputs[:, t, :] = out

        return outputs



