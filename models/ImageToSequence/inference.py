from ImageToSequenceModel import ImageToSequenceModel
from data_utils.ImageToSequence import ImageToSequenceDataset
from data_utils.misc_utils import label_tensor_to_char, clean_decoder_output

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

import os
import numpy as np
from PIL import Image
import fastwer

def infer(image, model):

    # transform image if image is not a torch tensor
    if isinstance(image, str):
        image = np.asarray(Image.open(image))
        transforms = T.Compose([T.ToPILImage(), T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = transforms(image)
        image = torch.unsqueeze(image, dim=0)

    image_embeddings = model.image_encoder(image)
    decoder_output = model.decoder(image_embeddings)

    decoder_output = label_tensor_to_char(decoder_output[0])
    return decoder_output

def evaluate(eval_set, model):
    # eval set is a torch dataset
    dataloader = DataLoader(eval_set, batch_size=1, shuffle=False)
    total_cer = 0

    for data in dataloader:
        image, label, image_name = data
        decoder_output = infer(image, model)

        decoder_output = clean_decoder_output(decoder_output)
        true_label = image_name[0].split('_')[0]
        cer = fastwer.score([true_label], [decoder_output], char_level=True)
        total_cer += cer
        print(true_label, decoder_output, cer, '\n')

    print('Average CER : {}'.format(total_cer/len(dataloader)))

if __name__ == '__main__':
    image_dir = '../../eval_data'
    eval_set = ImageToSequenceDataset(image_dir)

    model = ImageToSequenceModel(max_seq_length=18, image_embedding_dim=64)
    ckpt = '/Users/padmanabhankrishnamurthy/Downloads/img2seq_only_image_decoded_4000_0.0002944274456240237.pth'
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))

    # run evaluation
    # evaluate(eval_set, model)

    # run single image inference
    result = infer(os.path.join(image_dir, 'hey_2.jpg'), model)
    print(result)
    print(clean_decoder_output(result))