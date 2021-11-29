import torch
import torch.nn as nn
import torchvision.models as models

class ImageToSequenceModel(nn.Module):
    def __init__(self, max_seq_length, image_height, image_width, image_embedding_dim=512):
        '''

        :param max_seq_length: maximum word length, i.e, maximum number of characters in a word in the training data
        :param image_height: height of all input images
        :param image_width: width of all input images
        :param image_embedding_dim: embedding dimension of image encoder
        '''

        super(ImageToSequenceModel, self).__init__()
        self.max_seq_length = max_seq_length
        self.image_height = image_height
        self.image_width = image_width
        self.image_embedding_dim = image_embedding_dim

        self.image_encoder = self.get_image_encoder(self.image_height, self.image_width, self.image_embedding_dim)
        self.decoder = self.get_decoder(self.image_embedding_dim, self.max_seq_length)

    def get_image_encoder(self, height, width, embedding_dim):
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

    def get_decoder(self, image_embedding_dim, max_seq_length):
        '''
        :return: Decoder is just a linear layer that takes in the image embedding and the character sequence generated upto the current timestep, and predicts the next character
        '''
        input_size = image_embedding_dim + max_seq_length
        decoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=27), # 26 alphabets + <endToken>
            nn.Softmax()
        )
        return decoder



