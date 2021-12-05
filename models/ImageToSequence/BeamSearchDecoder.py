import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils.ImageToSequence import ImageToSequenceDataset
from models.ImageToSequence.ImageToSequenceModel import ImageToSequenceModel

import fastwer

class BeamNode():
    def __init__(self, parent, hidden_state, cell_state, output_char_embedding, seq_prob, seq, depth):
        self.parent = parent
        self.hidden_state = hidden_state
        self.cell_state = cell_state
        self.char_embedding = output_char_embedding
        self.seq_prob = seq_prob
        self.seq = seq
        self.depth = depth

class BeamDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, device='cpu'):
        super(BeamDecoder, self).__init__()

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

    def forward(self, features, beam_witdh=1, max_depth=18, teacher_forcing=False):

        # batch size
        batch_size = features.size(0)

        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(self.device)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(self.device)

        frontier = []
        hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
        output_char_index = torch.argmax(self.fc_out(hidden_state))
        output_char_prob = torch.max(self.fc_out(hidden_state))
        output_char_embedding = self.embed(output_char_index)
        root = BeamNode(None, hidden_state, cell_state, output_char_embedding, output_char_prob, [output_char_index], 0)
        frontier.append(root)

        depth = 0
        best_seq_prob = -1
        best_beam_node = None

        num_beams = 0

        while True:
            current = frontier.pop(0)

            hidden_state, cell_state = self.lstm_cell(torch.unsqueeze(current.char_embedding, dim=0), (current.hidden_state, current.cell_state))
            output = self.fc_out(hidden_state)

            topk_char_probs = torch.topk(output, beam_witdh).values[0]
            topk_char_indices = torch.topk(output, beam_witdh).indices[0]

            # add children
            if current.depth == max_depth:
                break

            for i in range(beam_witdh):
                num_beams+=1
                embedding = self.embed(topk_char_indices[i])
                seq_prob = current.seq_prob * topk_char_probs[i]
                current_seq = copy.deepcopy(current.seq)
                seq = current_seq + [topk_char_indices[i]]
                new_node = BeamNode(current, hidden_state, cell_state, embedding, seq_prob, seq, current.depth+1)
                frontier.append(new_node)
                print(num_beams)

                if seq_prob > best_seq_prob:
                    best_seq_prob = seq_prob
                    best_beam_node = new_node

            # print(num_beams/((3**18-1)/2), flush=True)

        return best_beam_node


if __name__ == '__main__':
    image_dir = '../../eval_data'
    dataset = ImageToSequenceDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ImageToSequenceModel(max_seq_length=18, image_embedding_dim=64)
    ckpt = '/Users/padmanabhankrishnamurthy/Downloads/img2seq_only_image_decoded_4000_0.0002944274456240237.pth'
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    # decoder = BeamDecoder(embed_size=64, hidden_size=512, vocab_size=29)

    total_cer = 0

    for data in dataloader:
        image, label, image_name = data
        image_embedding = model.image_encoder(image)
        decoder_output = model.decoder(image_embedding, label)

        decoder_output = dataset.label_tensor_to_char(decoder_output[0])
        # print(decoder_output)
        if '<end>' in decoder_output:
            decoder_output = decoder_output[decoder_output.index('<start>')+1:decoder_output.index('<end>')]
        else:
            decoder_output = decoder_output[decoder_output.index('<start>') + 1:decoder_output.index('<pad>')]
        decoder_output = ''.join(decoder_output)
        true_label = image_name[0].split('_')[0]
        cer = fastwer.score([true_label], [decoder_output], char_level=True)
        total_cer+=cer
        print(true_label, decoder_output, cer)
        print()

        # print(image_name, dataset.label_tensor_to_char(decoder_output[0]))
    print('Average CER : {}'.format(total_cer/len(dataloader)))