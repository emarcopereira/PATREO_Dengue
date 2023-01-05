import os
import PIL
import csv
import torch
import argparse
import torchvision
import io as IO
import numpy as np
import pandas as pd
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

    model_ft = models.resnet152(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    #checkpoint = torch.load('Network_balanced3.pt')
    #model_ft.load_state_dict(checkpoint['model_state_dict'])

    model_ft.load_state_dict(torch.load('Network_balanced3.pt'))

    return model_ft



def set_dataset():
    # use our dataset and defined transformations
    dataset_test = ICMDataset(root=root, mode="test", transforms=None)


    # define validation data loaders
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=4)

    # return (data_loader, data_loader_test)
    return data_loader_test


def go(model, data_loader_test, lr, epochs,  num_classes):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    test_metrics = list()

    paths = list()
    predictions = list()

    print('    Testing...')
    model.eval()

    with torch.no_grad():
        label_list = list()
        output_list = list()

        # Iterating over test batches.
        for it, data in enumerate(data_loader_test):
            # Obtaining images and labels for batch.
            inps, path = data
            # GPU casting. In CPU version comment the following line.
            inps = inps.cuda()
            # Forwarding inps through NN.
            output = model(inps)
            # Computing loss according to network prediction for batch and targets.

            # Getting labels and predictions from last epoch
            predictions += output.max(1)[1].cpu().numpy().tolist()
            paths += path



    predictions = np.asarray(predictions, dtype=np.int).ravel()
    paths = np.asarray(paths, dtype=np.str).ravel()

    return predictions, paths




#################### FAZ AS COISAS #####################
epochs = 10
num_classes = 3
lr = 1e-5
num_folds = 3
mod = 'resnet152'


root_path = '/mnt/DADOS_PARIS_1/ester/icm_aereo_classificacao/casa/test/'

model = get_classification_model(num_classes, mod)
data_loader = set_dataset()
preds, path = go(model, data_loader, lr, epochs,  num_classes)

