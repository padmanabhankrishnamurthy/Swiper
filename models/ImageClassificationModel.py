import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as T

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_utils.ImageClassification import ImageClassificationDataset


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(model, train_dataset, batch_size=4, save_path=None, save_name=None):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)

    epochs = 10
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader)

        for data in pbar:
            images, labels = data
            labels_class_indices = torch.argmax(labels, dim=2)
            labels_class_indices = torch.squeeze(labels_class_indices, dim=1)

            optimizer.zero_grad()
            pred = model(images.to(device))
            # print(pred.shape, train_dataset.unique_words[torch.argmax(labels).item()],  labels.shape, labels_class_indices.shape, labels_class_indices)
            loss = criterion(pred.to(device), labels_class_indices.to(device))
            loss.backward()
            optimizer.step()

            epoch_loss+=loss
            pbar.set_description('Epoch: {} Batch Loss: {}'.format(epoch+1, loss.item()/batch_size))

        if (epoch+1)%10==0:
            print('Epoch: {} | Loss: {}'.format(epoch+1, round(epoch_loss/len(train_dataset), 6)))
            if save_path and save_name:
                torch.save(model.state_dict(), '{}/{}_{}.pth'.format(save_path, save_name, epoch+1))



if __name__ == '__main__':
    images_dir = '../data'
    train_dataset = ImageClassificationDataset(images_dir)

    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=train_dataset.num_classes, bias=True)

    # freeze all params except final FC layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    train(model=model, train_dataset=train_dataset, save_path='../stored_models', save_name='classifier')



