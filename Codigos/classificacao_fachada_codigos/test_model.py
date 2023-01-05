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
from vit import get_timm_vit
from dataset import ICMDataset
from calculate_metrics import calculate_metrics, new_metrics
from torchvision.utils import save_image
from torch.distributions import Categorical

import os, json
from skimage.segmentation import mark_boundaries

from torchvision import models, transforms

from lime import lime_image
import pandas as pd
import random

from typing import Type, Any, Callable, Union, List, Optional, cast
from torch import Tensor
from collections import OrderedDict 

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
    
    elif rede == 'vit':
        model_ft = get_timm_vit(num_classes)

    else:
        model_ft = models.resnet152(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft


def get_imgs_with_labes(root, csv, n):
    csv = pd.read_csv(csv)

    imgs = os.listdir(root + 'images')

    cut = csv[csv['arquivo'].isin(imgs)].reset_index()
    cut = cut[['arquivo', 'label']]

    # cut = cut.groupby(['label']).count()
    # print(cut)
    files = list(cut['arquivo'])
    labels = list(cut['label'])
    return random.sample(files, n), random.sample(labels, n)

def set_dataset(bs):
    # use our dataset and defined transformations
    dataset = ICMDataset(root=root, mode="train", csv=csv_train, transforms=None)
    dataset_test = ICMDataset(root=root, mode="test", csv=csv_test, transforms=None)

    # Create a sampler by samples weights 
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=dataset.samples_weights,
        num_samples=dataset.__len__())


mod = 'resnet152'
num_classes = 3

model = get_classification_model(num_classes, mod)
PATH = "/mnt/DADOS_PARIS_1/joaopedro/fachadas2/models/resnet152_gamma_3bs12.pt"
model.load_state_dict(torch.load(PATH))
model.eval()

root = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/'
csv_train = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/fold/train2.csv'
csv_test = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/fold/test2.csv'

x = torch.rand([1,3,384,384])
#print(model)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

model.fc = Identity()
model.avgpool = Identity()

children_counter = 0
for n,c in model.named_children():
    print("Children Counter: ",children_counter," Layer Name: ",n,)
    children_counter+=1

y = model(x)
print(y.shape)


class NewModel(nn.Module):
    def __init__(self, model, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        #print(self.output_layers)
        self.selected_out = OrderedDict()
        #PRETRAINED MODEL
        self.pretrained = model
        self.fhooks = []

        for i,l in enumerate(list(self.pretrained._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.pretrained,l).register_forward_hook(self.forward_hook(l)))
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out

model2 = NewModel(model, output_layers = [4,5,6,7])

y, selected = model2(x)
print(selected['layer3'].shape)
