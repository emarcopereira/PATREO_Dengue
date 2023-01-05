import torch
import matplotlib.pyplot as plt
import numpy as np
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

gpu = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
#device = torch.device("cuda")
device = torch.device("cpu")
num_classes = 9
H = 24
dim = 768
depth = 1
num_heads = 2
mlp_dim = 3072


downsample = transforms.Resize((64,64))
upsample = transforms.Resize((224,224))
img2pil = transforms.ToPILImage()


files = os.listdir('/mnt/DADOS_PARIS_1/ester/cropped_dataset/images/')
files = random.sample(files, 10)

print(files)
for file in files:

    img = io.imread(f"/mnt/DADOS_PARIS_1/ester/cropped_dataset/images/"+file)
    gt = io.imread(f"/mnt/DADOS_PARIS_1/ester/cropped_dataset/masks/"+file)
    img = img.astype(np.float32)/255.0
    gt = gt.astype(np.int64)
    unique, count = np.unique(gt, return_counts=True)
    print("first count")
    print(unique, count)


    #img = cv.imread(f"/mnt/DADOS_PARIS_1/joaopedro/crop_9873_img.tiff")
    #gt = Image.open(f"/mnt/DADOS_PARIS_1/joaopedro/crop_9873.tiff")


    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    print(gt.shape)
    #gt = downsample(gt)
    img = resize(img, (224,224), anti_aliasing=True)
    #gt = rescale(gt, 0.285714286, anti_aliasing=False)
    gt = downscale_local_mean(gt, (4, 4))
    print(gt.shape)
    #gt = to_tensor_target_lc(gt)
    #gt = gt.reshape(4096)
    gt = gt.reshape(56*56)
    print(gt.size)
    gt = gt.astype(np.int64)
    unique, count = np.unique(gt, return_counts=True)
    print(unique, count)

    break
    # define the transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])   

    img = np.array(img)
    # apply the transforms
    img = transform(img)

    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)

    # PREPARE MODEL
    if model_name == "fcn":
        model = torch.load('/mnt/DADOS_PARIS_1/matheusp/dengue/chessmix/models/fcn-resnet101/fcn-resnet101_best.pth')

    print("loaded state dict")
    model.to(device)
    # !!! DONT FORGET TO SET MODEL TO EVAL MODE !!!
    model.eval()

    features = model.backbone(img)

    features = features['out']   #B x C x H/8 x W/8
    if model_name == 'setr':
        features = upsample(features)

    print(features.shape)
    up = nn.Upsample(scale_factor=2, mode='nearest')
    features = up(features).squeeze(0)
    print(features.shape)
    features = features.reshape(2048, 56*56)
    features = features.permute(1,0).contiguous()
    features = features.detach().numpy()

    tsne = TSNE(2, perplexity=30, learning_rate=1000, n_iter=4000,verbose=1, init='pca')
    tsne_proj = tsne.fit_transform(features)
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20b')
    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 9
    for lab in classes:
        print(lab)
        indices = gt==classes_id[lab]
        #ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=colors_per_class[lab], label = lab ,alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.show()

    plt.savefig('tsne_'+file+'.png')