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
from torch.distributions import Categorical
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
from skimage.segmentation import mark_boundaries
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from lime import lime_image
import pandas as pd
import random
from sklearn.utils import shuffle



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


def get_imgs_with_labes(root, n):

    files1 = os.listdir(root + 'classe1')
    imgs_classe1 = [os.path.join(root + 'classe1', i) for i in files1]
    labels1 = [1] * len(imgs_classe1)

    files2 = os.listdir(root + 'classe2')
    imgs_classe2 = [os.path.join(root + 'classe2', i) for i in files2]
    labels2 = [2] * len(imgs_classe2)

    files3 = os.listdir(root + 'classe3')
    imgs_classe3 = [os.path.join(root + 'classe3', i) for i in files3]
    labels3 = [3] * len(imgs_classe3)

    imgs = imgs_classe1 + imgs_classe2 + imgs_classe3
    labels = labels1 + labels2 + labels3

    imgs, labels = shuffle(imgs, labels)

    return imgs[0:n], labels[0:n]


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 

# resize and take the center part of image to what our model expects
def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])       
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])    

    return transf


def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(512)
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf  

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(512)
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf    

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

def get_device(gpu):
    device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    print("device used:",device)
    return device

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = get_device(0)
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

mod = 'resnet152'
num_classes = 3

model = get_classification_model(num_classes, mod)
PATH = '/mnt/DADOS_PARIS_1/ester/classificacao_aerea/Network_quintal.pt'
model.load_state_dict(torch.load(PATH))
model.eval()

root = '/mnt/DADOS_PARIS_1/ester/icm_aereo_classificacao/quintal/test/'



files, labels = get_imgs_with_labes(root, 30)
print(files)
print(labels)

device = get_device(0)

for img_path, label in zip(files, labels):
    print(img_path)
    img = get_image(img_path)
    img_t = get_input_tensors(img)
    logits = model(img_t)
    probs = F.softmax(logits, dim=1)
    probs1 = probs.topk(1)
    classe_pred = np.argmax(probs.detach().numpy()) + 1
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pill_transf(img)), 
                                             batch_predict, # classification function
                                             top_labels=1, 
                                             hide_color=0, 
                                             num_samples=30) # number of images that will be sent to classification function
    model.to('cpu')
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    plt.title('classe predita: ' + str(classe_pred) + ' - classe real: ' + str(label))
    plt.imshow(img_boundry1)
    plt.savefig('/mnt/DADOS_PARIS_1/ester/classificacao_aerea/lime_images/quintal/'+img_path.split('/')[-1]+'_lime.jpg')


    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    img_boundry2 = mark_boundaries(temp/255.0, mask)
    plt.title('classe predita: ' + str(classe_pred) + ' - classe real: ' + str(label))
    plt.imshow(img_boundry2)
    plt.savefig('/mnt/DADOS_PARIS_1/ester/classificacao_aerea/lime_images/quintal/'+img_path.split('/')[-1]+'_lime2.jpg')