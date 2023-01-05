import os
import PIL
import csv
import torch
import utils
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

parser = argparse.ArgumentParser()
parser.add_argument('--img_type', type=str, required=True)
args = parser.parse_args()

root_path = "/mnt/DADOS_PONTOISE_1/matheusp/datasets/icm/"
img_type = args.img_type # aerial / street

class ICMDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transforms=None):
        self.root = root
        self.transforms = transforms
        self.mode = mode

        self.imgs = []
        self.imgs_class = [[] for x in range(7)]
        self.ok_list = []

        self.ok = csv.reader(open(os.path.join(root, "dataset_final/ok.txt"), 'r'))
        for row in self.ok:
            self.ok_list += row

        with IO.open(os.path.join(root, "data_processed.csv"), "r", encoding="UTF-8") as file:
            self.reader = np.genfromtxt(file, delimiter=",", dtype=None, encoding=None)

        self.dicio = {}

        # Gera dicionario com info de cada imagem
        for row in self.reader[1:]:
            # print(row)
            k = row[0]
            v = row[1:]
            if k in self.ok_list:
                self.dicio[k] = v
                img_name = str(k)
                self.imgs_class[int(v[3]) - 3] += [img_name]

        for i in range(7):
            self.imgs_class[i] = np.array(sorted(self.imgs_class[i]))

        # Divide dataset em treino e teste
        np.random.seed(42)
        perm = [np.random.permutation(len(self.imgs_class[i])) for i in range(7)]

        if self.mode == "train":
            for i in range(7):
                self.imgs.extend(self.imgs_class[i][perm[i][:int(0.8 * perm[i].shape[0])]])
        elif self.mode == "test":
            for i in range(7):
                self.imgs.extend(self.imgs_class[i][perm[i][int(0.8 * perm[i].shape[0]):]])

        # print(self.imgs)


    def __getitem__(self, idx):
        # Carrega image e label
        # print("img_data: ")
        img_id = self.imgs[idx].split("_")[0]
        img_data = self.dicio[img_id]
        # print(img_data)
        img_name = str(img_id) + ".png"

        img_path = os.path.join(self.root, "dataset_final/" + img_type, img_name)
        img = Image.open(img_path).convert("RGB")

        # Pega label do dicionario
        label = img_data[3]
        label = int(label) - 3
        # print(img_path, label)

        # Transforma imagem em tensor
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


# Pega model de classificacao
def get_classification_model(img_type, num_classes):
    model_ft = None

    model_ft = models.vgg11_bn(pretrained=True)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    input_size = 224

    return model_ft


def get_transform(train):
    transforms = []
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def set_dataset():
    # use our dataset and defined transformations
    dataset = ICMDataset(root=root_path, mode="train", transforms=get_transform(train=True))
    dataset_test = ICMDataset(root=root_path, mode="test", transforms=get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=4)

    # return (data_loader, data_loader_test)
    return data_loader, data_loader_test


def go(model, data_loader, data_loader_test, lr, epochs):
    print(torch.cuda.is_available())
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    # print(model)

    class_weights=torch.FloatTensor([1.655, 5.938, 1, 2.4, 2.028, 9.6, 82.285]).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_metrics = list()
    test_metrics = list()

    model.eval()

    labels = list()
    predictions = list()

    batch_metrics_train = np.array([])
    batch_metrics_test = np.array([])

    with torch.no_grad():
        label_list = list()
        output_list = list()

        # Iterating over test batches.
        for it, data in enumerate(data_loader_test):
            # Obtaining images and labels for batch.
            inps, labs = data
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

        test_metrics.append(np.mean(batch_metrics_test))

        label_array = np.asarray(label_list, dtype=np.int).ravel()
        output_array = np.asarray(output_list, dtype=np.int).ravel()

        # print('Epoch: %d, Accuracy: %.2f%%' % (ep + 1, 100.0 * np.sum(label_array == output_array) / float(label_array.shape[0])))

    # Save stuff
    labels = np.asarray(labels, dtype=np.int).ravel()
    predictions = np.asarray(predictions, dtype=np.int).ravel()
    dump(labels, "test_"+ img_type + "/labels_balanced_" + img_type + ".joblib")
    dump(predictions, "test_"+ img_type + "/predictions_balanced_" + img_type + ".joblib")
    # torch.save(model.state_dict(), "Network_balanced_" + img_type + ".pt")

    # Faz graficos
    # Transforming list into ndarray for plotting.
    training_array = np.asarray(training_metrics, dtype=np.float32)
    test_array = np.asarray(test_metrics, dtype=np.float32)

    # Plotting error metric.
    fig, ax = plt.subplots(1, 2, figsize = (16, 8), sharex=False, sharey=True)

    ax[0].plot(training_array)
    ax[0].set_xlabel('Training Loss Progression')

    ax[1].plot(test_array)
    ax[1].set_xlabel('Test Loss Progression')

    plt.savefig("charts_balanced_" + img_type + ".png")


#################### FAZ AS COISAS #####################
epochs = 10
num_classes = 7
lr = 0.0
if img_type == "aerial":
    lr = 1e-3
elif img_type == "street":
    lr = 1e-5

model = get_classification_model(img_type, num_classes)
model = torch.load('/home/users/eduardo/ICM/exaust_balanced/results_' + img_type + '/vgg11_0.0001_adam/vgg11_final_model_ft')
data_loader, dataset = set_dataset()
go(model, data_loader, dataset, lr, epochs)
