import numpy as np
from torch.utils import data
import pandas as pd
import os
import torch
from skimage.io import imread
from skimage import transform
import skimage.transform as scikit_transform
from skimage.transform import rotate, AffineTransform
from PIL import Image
import random
from skimage import transform
from skimage.util import random_noise
from skimage.util import crop
from skimage.color import rgb2gray
import cv2

#def make_dataset(root, csv):
#    csv = pd.read_csv(csv)
#
#    imgsFolders = os.listdir(root + 'arquivos')
#    imgs = []
#    for folder in imgsFolders:
#        imgs_in_folder = os.listdir(root + 'arquivos/' + folder)
#        for img in imgs_in_folder:
#            imgs.append('arquivos/' + folder +'/' + img)
#    
#    #print(imgs)
#
#    cut = csv[csv['arquivo'].isin(imgs)].reset_index()
#    cut = cut[['arquivo', 'label']]
#    cut2 = cut.groupby(['label']).count()
#    print(cut2)
#
#    return list(cut['arquivo']), list(cut['label'])

def make_dataset(root, csv):
    csv = pd.read_csv(csv)

    imgs = os.listdir(root + 'images')

    cut = csv[csv['arquivo'].isin(imgs)].reset_index()
    cut = cut[['arquivo', 'label']]
    cut = cut[cut['label'] != 2]
    cut['label'] = cut['label'].replace(3, 2)
    cut2 = cut.groupby(['label']).count()
    #print(cut2)

    return list(cut['arquivo']), list(cut['label'])



class ICMDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, csv, vit = False, transforms=None):
        self.root = root
        self.mode = mode
        self.csv = csv
        self.vit = vit
        self.imgs, self.labels = make_dataset(self.root, self.csv)
        #print(len(self.imgs))
        self.labels = [i - 1 for i in self.labels]
        #print(np.unique(self.labels))
        #diminuir de 5 para 3 classes
        #self.labels = [0 if i==1 else i for i in self.labels]
        #self.labels = [1 if i==2 else i for i in self.labels]
        #self.labels = [2 if i==3 else i for i in self.labels]
        #self.labels = [2 if i==4 else i for i in self.labels]

        if mode == 'train':
            self.weight_class = np.unique(np.array(self.labels), return_counts=True)[1]
            self.samples_weights = self.weight_class[self.labels]
        else:
            self.weight_class = None
            self.samples_weights = None

        self.transforms = transforms


    def __getitem__(self, idx):
        #img_path = self.root  + self.imgs[idx]
        img_path = self.root + 'images/' + self.imgs[idx]

        lbl = self.labels[idx]

        img = imread(img_path)

        img = (img-img.mean())/img.std()
        img = ((img - np.min(img)) / (np.max(img) - np.min(img)))

        if img.shape[0] != 384 or img.shape[1] != 512:
            img = scikit_transform.resize(img, (384, 512)).astype(img.dtype)

        if self.vit:
            pass
            #img = img[224:384-224, 224:512-224]
            #img = crop(img, ((224, 512-224), (224, 384-224), (0,0)), copy=False)
            img = scikit_transform.resize(img, (224, 224)).astype(img.dtype)

        if self.transforms is not None:
            img = self.transforms(img)
        
        W = img.shape[1]
        H = img.shape[0]
        C = 3
        #grayscale
        #img = rgb2gray(img)
        #C = 1

        #data augmentation
        if self.mode == 'train':
            aug_choice = np.random.randint(6)
            if aug_choice == 0:
                #Flip an array horizontally.
                img = np.fliplr(img).copy()
                
            elif aug_choice == 1:
                #Flip an array horizontally.
                img = np.flipud(img).copy()
            
            elif aug_choice == 2:
                angle = (np.random.rand(1) - 0.5) * 20
                img = transform.rotate(img, angle)

            elif aug_choice == 3:
                img = random_noise(img)

            elif aug_choice == 4:
                img = img + (100/255)

            else:
                tf = AffineTransform(shear=-0.5)
                img = transform.warp(img, tf, order=1, preserve_range=True, mode='wrap')

            #adicionar augmentation que muda cor para ver problema do vies do muro

        #em pytorch: (C, H, W)
        #skimage: (H, W, C)
        #img = img[:,:, np.newaxis]
        #img = img.reshape(C,H,W)
        img = torch.from_numpy(img)
        #img = img.view(img.shape[2], img.shape[0], img.shape[1])
        img = img.permute(2,0,1)
        #print(img.shape)
        img = img.float()


        return img, lbl, img_path


    def __len__(self):
        return len(self.imgs)


root = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/'
csv_train = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/fold/train2.csv'
csv_test = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/fold/test2.csv'
#root = '/mnt/DADOS_PARIS_1/matheusp/csv_segunda_leva/'
#csv_train = '/mnt/DADOS_PARIS_1/joaopedro/fachadas2/folds/train2.csv'
#csv_test = '/mnt/DADOS_PARIS_1/joaopedro/fachadas2/folds/test2.csv'


#dataset_train = ICMDataset(root, 'train', csv_train)
#dataset_test = ICMDataset(root, 'test', csv_test)


