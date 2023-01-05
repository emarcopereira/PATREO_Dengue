import os
import PIL
import csv
import torch
import argparse
import torchvision
import io as IO
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from joblib import dump, load
from sklearn import datasets
from skimage import io
from matplotlib import pyplot as plt
from torchvision import models
from torch.utils import data
from torch.utils.data import DataLoader

from dataset import ICMDataset
from calculate_metrics import calculate_metrics, new_metrics
from torchvision.utils import save_image


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Pega model de classificacao
def get_classification_model(num_classes, rede, feature_extract=False, use_pretrained=True):
    model_ft = None

    if rede == 'resnet50':
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif rede == 'densenet121':
        model_ft = models.densenet121(pretrained=use_pretrained)
        model_ft.features.conv0 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    else:
        model_ft = models.resnet152(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft



def set_dataset():
    # use our dataset and defined transformations
    dataset = ICMDataset(root=root_train, mode="train", transforms=None)
    dataset_test = ICMDataset(root=root_test, mode="test", transforms=None)

    # Create a sampler by samples weights 
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=dataset.samples_weights,
        num_samples=dataset.__len__())


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=4)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=4)


    return data_loader, data_loader_test


def go(model, data_loader, data_loader_test, lr, epochs, num_classes):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    class_weights = torch.FloatTensor([0.98, 0.8, 0.22]).cuda() #telhado
    class_weights = torch.FloatTensor([0.95, 0.75, 0.3]).cuda() #quintal
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    #criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_metrics = list()
    test_metrics = list()

    epochs = epochs
    for ep in range(epochs):
        labels = list()
        predictions = list()
        incorrect_path = list()

        print('##############################################')
        print('Starting epoch ' + str(ep + 1) + '/' + str(epochs) + '...')

        # Setting model to training mode.
        print('    Training...')
        model.train()

        batch_metrics_train = np.array([])
        batch_metrics_test = np.array([])

        # Iterating over training batches.
        for it, data in enumerate(data_loader):
            # Obtaining data and labels for batch.
            inps, labs, img_path = data
            #save_image(inps, 'teste/{}.png'.format(it))
            # GPU casting. In CPU version comment the following two lines.
            inps = inps.cuda()
            labs = labs.cuda()
            # Zeroing optimizer.
            optimizer.zero_grad()
            # Forwarding inps through NN.
            output = model(inps)
            # Computing loss according to network prediction for batch and targets.
            loss = criterion(output, labs)
            # Backpropagating loss.
            loss.backward() # All backward pass is computed from this line automatically by package torch.autograd.
            # Taking optimization step (updating NN weights).
            optimizer.step()
            # Appending metric for batch.
            batch_metrics_train = np.append(batch_metrics_train, loss.data.item())

        # Setting model to evaluation mode.
        training_metrics.append(np.mean(batch_metrics_train))
        print('    Testing...')
        model.eval()

        with torch.no_grad():
            label_list = list()
            output_list = list()

            # Iterating over test batches.
            for it, data in enumerate(data_loader_test):
                # Obtaining images and labels for batch.
                inps, labs, img_path = data

                # GPU casting. In CPU version comment the following line.
                inps = inps.cuda()
                labs = labs.cuda()
                # Forwarding inps through NN.
                output = model(inps)
                # Computing loss according to network prediction for batch and targets.
                loss = criterion(output, labs)
                # Appending metric for batch.
                batch_metrics_test = np.append(batch_metrics_test, loss.data.item())

                # Getting labels and predictions from last epoch.
                label_list += labs.cpu().numpy().tolist()
                output_list += output.max(1)[1].cpu().numpy().tolist()
                labels += labs.cpu().numpy().tolist()
                predictions += output.max(1)[1].cpu().numpy().tolist()
                for i in img_path:
                	incorrect_path.append(i)
               
                # if ep == 0:
                #     for idx, k in enumerate(img_path):
                #         incorrect_path.append(str(img_path[idx]) + ' ' + str(labels[idx]) + ' ' + str(predictions[idx]))


            test_metrics.append(np.mean(batch_metrics_test))

            label_array = np.asarray(label_list, dtype=np.int).ravel()
            output_array = np.asarray(output_list, dtype=np.int).ravel()
            img_array = np.asarray(incorrect_path).ravel()

            calculate_metrics(output_array, label_array)

            if ep == epochs-1:

                for i in range(len(img_array)):
                    print(str(img_array[i]) + ' ' + str(label_array[i]) + ' ' + str(output_array[i]), file=open('ultimo2.txt', "a"))


    # Save stuff
    labels = np.asarray(labels, dtype=np.int).ravel()
    predictions = np.asarray(predictions, dtype=np.int).ravel()
    torch.save(model.state_dict(), "Network_pav_quintal" + ".pt")
    # model.load_state_dict(torch.load('Densenet169.pt'))

    # Faz graficos
    # Transforming list into ndarray for plotting.
    training_array = np.asarray(training_metrics, dtype=np.float32)
    test_array = np.asarray(test_metrics, dtype=np.float32)

    # Plotting error metric.
    # fig, ax = plt.subplots(1, 2, figsize = (16, 8), sharex=False, sharey=True)

    # ax[0].plot(training_array)
    # ax[0].set_xlabel('Training Loss Progression')

    # ax[1].plot(test_array)
    # ax[1].set_xlabel('Test Loss Progression')

    # plt.savefig("charts_balanced3" + ".png")



#################### FAZ AS COISAS #####################
epochs = 10
num_classes = 3
lr = 1e-5
mod = 'resnet152'


root_train = '/mnt/DADOS_PARIS_1/ester/icm_aereo_classificacao/pav_quintal/train/'
root_test = '/mnt/DADOS_PARIS_1/ester/icm_aereo_classificacao/pav_quintal/test/'

model = get_classification_model(num_classes, mod)
data_loader, dataset = set_dataset()
go(model, data_loader, dataset, lr, epochs, num_classes)
