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
import cv2
from sklearn.utils import shuffle

def make_dataset(root):

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

    return imgs, labels




class ICMDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transforms=None):
        self.root = root
        self.mode = mode

        self.imgs, self.labels = make_dataset(self.root)
        self.labels = [i - 1 for i in self.labels]

        if mode == 'train':
            self.weight_class = np.unique(np.array(self.labels), return_counts=True)[1]
            self.samples_weights = self.weight_class[self.labels]
        else:
            self.weight_class = None
            self.samples_weights = None

        self.transforms = transforms


    def __getitem__(self, idx):

        img_path = self.imgs[idx]


        lbl = self.labels[idx]

        img = imread(img_path)

        img = (img-img.mean())/img.std()
        img = ((img - np.min(img)) / (np.max(img) - np.min(img)))

        if img.shape[0] != 200 or img.shape[1] != 200:
             img = scikit_transform.resize(img, (200, 200)).astype(img.dtype)


        if self.transforms is not None:
            img = self.transforms(img)

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

        img = torch.from_numpy(img)
        img = img.permute(2,1,0)
        img = img.float()


        return img, lbl, img_path


    def __len__(self):
        return len(self.imgs)


# root_train = '/mnt/DADOS_PARIS_1/ester/icm_aereo_classificacao/casa/train/'
# root_test = '/mnt/DADOS_PARIS_1/ester/icm_aereo_classificacao/casa/test/'


# dataset_train = ICMDataset(root_train, 'train')
# dataset_test = ICMDataset(root_test, 'test')

# print(dataset_train.__len__())


