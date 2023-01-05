import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import cv2 as cv
import argparse
from torchvision import models, transforms
from torch.utils import data
from torchvision.models.segmentation import fcn_resnet101, fcn_resnet50
import os
import PIL
from PIL import Image
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
import random

model_name = "fcn"

colors_per_class = {
    0 : [0, 0, 0],
    1 : [255, 107, 107],
    2 : [10, 189, 227],
    3 : [25, 60, 243],
    4 : [156, 72, 132]
}
colors_per_class = {
    'asfalto' : 'orange',
    'arvore' : 'maroon',
    'cimento' : 'green',
    'telhado1' : 'blue',
    'telhado2' : 'yellow',
    'telhado3': 'cyan',
    'grama' : 'pink',
    'piscina': 'gray',
    'solo': 'black'
}

classes = ['asfalto', 'arvore', 'cimento', 'telhado1', 'telhado2', 'telhado3', 'grama', 'piscina', 'solo']
classes_id = {
    'asfalto' : 0,
    'arvore'  : 1,
    'cimento' : 2,
    'telhado1': 3,
    'telhado2': 4,
    'telhado3': 5,
    'grama'   : 6,
    'piscina' : 7,
    'solo'    : 8
}

def to_tensor_target_lc(mask):
    # For the landcoverdataset
    mask = np.array(mask)
    mask = np.mean(mask) 

    return torch.LongTensor(mask)


folders = os.listdir('/mnt/DADOS_PARIS_1/ester/dataset_fusion/aerea/casa_msks/train')
#files = random.sample(files, 10)
files1 = os.listdir('/mnt/DADOS_PARIS_1/ester/dataset_fusion/aerea/casa_msks/train/classe1')
files2 = os.listdir('/mnt/DADOS_PARIS_1/ester/dataset_fusion/aerea/casa_msks/train/classe2')
files3 = os.listdir('/mnt/DADOS_PARIS_1/ester/dataset_fusion/aerea/casa_msks/train/classe3')

all_files = [files1, files2, files3]

ICM1 = {
    'asfalto' : np.zeros(300),
    'arvore'  : np.zeros(300),
    'cimento' : np.zeros(300),
    'telhado1': np.zeros(300),
    'telhado2': np.zeros(300),
    'telhado3': np.zeros(300),
    'grama'   : np.zeros(300),
    'piscina' : np.zeros(300),
    'solo'    : np.zeros(300)
}

ICM2 = {
    'asfalto' : np.zeros(300),
    'arvore'  : np.zeros(300),
    'cimento' : np.zeros(300),
    'telhado1': np.zeros(300),
    'telhado2': np.zeros(300),
    'telhado3': np.zeros(300),
    'grama'   : np.zeros(300),
    'piscina' : np.zeros(300),
    'solo'    : np.zeros(300)
}

ICM3 = {
    'asfalto' : np.zeros(300),
    'arvore'  : np.zeros(300),
    'cimento' : np.zeros(300),
    'telhado1': np.zeros(300),
    'telhado2': np.zeros(300),
    'telhado3': np.zeros(300),
    'grama'   : np.zeros(300),
    'piscina' : np.zeros(300),
    'solo'    : np.zeros(300)
}


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx]

def count_classes (root):

    X = np.zeros([900,10])

    index = 0
    for classe in [1,2,3]:
        files = os.listdir(root+"classe"+str(classe)+"/")
        print(len(files))
        files = files[:300]
        for file in files:
            X[index][9] = classe
            #print(file)
            prds = np.load(root+"classe"+str(classe)+"/"+file)
            prds_crop = crop_center(prds,50,50)         
            #prds = prds.max(0)
            #prds = prds.astype(int)
            #unique, count = np.unique(prds, return_counts=True)
            #print(unique, count)
            #print(prds)
            h, w = prds_crop.shape
        
            new = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    #if (prds[i][j] < 9):
                    X[index][prds_crop[i][j]] += 1
                    

            index += 1

    return X

icms = ["classe1", "classe2", "classe3"]

root = '/mnt/DADOS_PARIS_1/matheusp/dengue/chessmix/forward_outputs/fcn-resnet101/probs_casa2/'

X = count_classes(root)
df = pd.DataFrame(X)

df.to_csv("stats_crop.csv")