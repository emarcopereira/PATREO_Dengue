import os
import sys
import time
import numpy as np
import argparse
import copy
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision.models.segmentation import fcn_resnet101, fcn_resnet50
from torchvision import transforms
import skimage.io as io
import seaborn as sns
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

import dataset
import evaluation

#python3 main.py --dataset ../vaihingen_1000/ --epochs 10 --batch_size 8 --num_workers 4 --n_classes 6 --lr 1e-4

parser = argparse.ArgumentParser(description='PyTorch FCN')
parser.add_argument('--dataset', type=str, default='grid_campinas_inteira', help="Name of the dataset")
parser.add_argument('--network', type=str, default='resnet101', help="Name of the network (resnet101|resnet50)")
parser.add_argument('--exp_name', type=str, default='fcn-resnet101', help="Name of the experiment")
parser.add_argument('--model_path', type=str, default='models/fcn-resnet101/fcn-resnet101_best.pth', help="Path to the pretrained model")
parser.add_argument('--fold', type=int, default=None, help="Fold number")
parser.add_argument('--n_classes', type=int, default=9, help="number of classes")
parser.add_argument('--save_imgs', type=str, default='True', help="Save or not the resulting thematic maps (True|False)")

args = parser.parse_args()

def normalize_rows(array):
    sum = array.sum(axis=1)
    new = np.zeros(array.shape)
    for i in range(array.shape[0]):
        new[i] = array[i]/sum[i]
    return new


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    


def test(test_loader, net):

    tic = time.time()
    
    # Setting network for evaluation mode.
    net.eval()

    # Lists for metrics.
    #int_all = np.asarray(args.n_classes, dtype=np.float32)
    #uni_all = np.asarray(args.n_classes, dtype=np.float32)
    int_all = np.asarray(0, dtype=np.float32)
    uni_all = np.asarray(0, dtype=np.float32)

    check_mkdir('forward_outputs/'+args.exp_name+'/probs_campinas_inteira')
    gts_all = []
    preds_all = []
    # Iterating over batches.
    for i, batch_data in enumerate(test_loader):
        #print('Validation: epoch {}, iteration {}'.format(epoch, i))
        # Obtaining images and labels for batch.
        inps, img_name = batch_data

        # Casting to cuda variables.
        inps = inps.to(device)

        # Forwarding through network.
        outs = net(inps)
        outs = outs['out']

        # Obtaining predictions.
        #print(outs.shape)
        prds = outs.data.max(1)[1].squeeze_(1).squeeze(0).cpu().numpy()
        
        #prds = outs.data.squeeze().cpu().numpy()
        #print(np.unique(prds))
        
        np.save('forward_outputs/'+args.exp_name+'/probs_campinas_inteira/'+img_name[0].replace('.tif', '.npy').replace('test_', ''), prds)
        #preds_all.append(prds)
        
        # if args.save_imgs == 'True':
        #     h, w = prds.shape
        
        #     new = np.zeros((h, w, 3), dtype=np.uint8)
        #     for i in range(h):
        #         for j in range(w):
        #             #if label[i][j] == -1:
        #             #    new[i][j] = [0,0,0]

        #             if prds[i][j] == 0: # asfalto
        #                 new[i][j] = [255,0,0]

        #             elif prds[i][j] == 1: # arvore
        #                 new[i][j] = [0,255,0]

        #             elif prds[i][j] == 2: # cimento
        #                 new[i][j] = [0,0,255]

        #             elif prds[i][j] == 3: # telhado 1
        #                 new[i][j] = [255,255,0]

        #             elif prds[i][j] == 4: # telhado 2
        #                 new[i][j] = [0,255,255]

        #             elif prds[i][j] == 5: # telhado 3
        #                 new[i][j] = [255, 0, 255]

        #             elif prds[i][j] == 6: # grama
        #                 new[i][j] = [153,0,76]

        #             elif prds[i][j] == 7: # piscina
        #                 new[i][j] = [0,153,76]

        #             elif prds[i][j] == 8: # solo exposto
        #                 new[i][j] = [153,0,153]

        #     io.imsave('forward_outputs/'+args.exp_name+'/sombra_quintal/classe3/'+img_name[0].replace('.tif', '.png'), new)

    toc = time.time()


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)
#if args.network == 'resnet101':
#    model = fcn_resnet101(num_classes=args.n_classes)
#else:
#    model = fcn_resnet50(num_classes=args.n_classes)
model = torch.load(args.model_path)
#print(pth)
#model.load_state_dict(pth)
model = model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

test_set = dataset.ListDataset_forward(mode='test', dataset=args.dataset, fold=args.fold, new_data_size=0, transform=transform)

print('Test size: ', len(test_set))

test_loader = DataLoader(test_set,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False)

# Iterating over epochs.
test(test_loader, model)

    
