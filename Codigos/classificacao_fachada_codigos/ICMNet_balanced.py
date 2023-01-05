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
import tqdm
import copy

def get_device(gpu):
    device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    print("device used:",device)
    return device

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' \
               else torch.sum(loss) if self.reduction == 'sum' \
               else loss

class GSC_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(GSC_Loss, self).__init__()
    
    def scaled_entropy (self, inputs):
        E = Categorical(inputs).entropy()
        return E/(len(inputs))

    def normpdf(self, x, mean, sd):
        pi = 3.1415927410125732
        var = float(sd)**2
        denom = (2*pi*var)**.5
        num = torch.exp(-(x-mean)**2/(2*var))
        return num/denom

    def joint_normal(self, K, mu, sigma):
        f = 1
        for i in range(len(K)):
            f = f * self.normpdf(K[i], mu, sigma)     
        return f

    def forward(self, inputs, targets, topc=2, gama = 0.2, smooth=1):
        
        CE = F.cross_entropy(inputs, targets, reduction='mean')
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.softmax(inputs, dim=-1)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        GSC = self.joint_normal(torch.topk(inputs, topc).values, 0, self.scaled_entropy(inputs))
        loss = CE + gama*GSC
        
        return loss

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
    
    elif rede == 'vit':
        model_ft = get_timm_vit(num_classes)

    else:
        model_ft = models.resnet152(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft



def set_dataset(bs, vit):
    # use our dataset and defined transformations
    dataset = ICMDataset(root=root, mode="train", csv=csv_train, vit = vit, transforms=None)
    dataset_test = ICMDataset(root=root, mode="test", csv=csv_test, vit = vit, transforms=None)

    # Create a sampler by samples weights 
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=dataset.samples_weights,
        num_samples=dataset.__len__())


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=4)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=bs, shuffle=False, num_workers=4)

    # return (data_loader, data_loader_test)
    return data_loader, data_loader_test, dataset.weight_class


def go(model, data_loader, data_loader_test, lr, epochs, weights, num_classes, gpu, model_name):

    device = get_device(gpu)

    model.to(device)

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_size = len(data_loader_test)

    #class_weights = torch.FloatTensor([0.82, 0.69, 0.49]).cuda() #pesos para todos os ids
    #class_weights = torch.FloatTensor([0.8, 0.67, 0.53]).cuda() #pesos so para id=1
    #class_weights = torch.FloatTensor([0.978, 0.84, 0.69, 0.63, 0.86]).cuda() #pesos para 5 classes
    class_weights = torch.FloatTensor([0.75, 0.69, 0.56]).to(device) #pesos com dados zooniverse
    #[0.8092517006802721, 0.6835374149659864, 0.5072108843537415]
    #[0.40462585034013604, 0.3417687074829932, 0.25360544217687075]

    #class_weights = torch.FloatTensor([0.8092517006802721, 0.6835374149659864, 0.5072108843537415]).to(device)
    class_weights = torch.FloatTensor([0.65, 0.35]).to(device)
    #criterion = nn.CrossEntropyLoss(weight = class_weights).to(device)
    #criterion_name = "CE"
    #criterion = nn.CrossEntropyLoss().to(device)
    gamma = 2
    criterion = FocalLoss(gamma=gamma, alpha = class_weights, reduction='mean').to(device)
    criterion_name = "fl"

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
        running_loss = 0.0
        running_corrects = 0
        # Iterating over training batches.
        for it, data in enumerate(data_loader):
            # Obtaining data and labels for batch.
            inps, labs, img_path = data
            #print(labs.data)
            #save_image(inps, 'teste/{}.png'.format(it))
            # GPU casting. In CPU version comment the following two lines.
            inps = inps.to(device)
            labs = labs.to(device)
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

            running_corrects = 0
            # Iterating over test batches.
            for it, data in enumerate(data_loader_test):
                # Obtaining images and labels for batch.
                inps, labs, img_path = data

                # GPU casting. In CPU version comment the following line.
                inps = inps.to(device)
                labs = labs.to(device)
                # Forwarding inps through NN.
                output = model(inps)
                _, preds = torch.max(output, 1)
                # Computing loss according to network prediction for batch and targets.
                loss = criterion(output, labs)
                # Appending metric for batch.
                batch_metrics_test = np.append(batch_metrics_test, loss.data.item())

                # Getting labels and predictions from last epoch.
                label_list += labs.cpu().numpy().tolist()
                output_list += output.max(1)[1].cpu().numpy().tolist()
                labels += labs.cpu().numpy().tolist()
                predictions += output.max(1)[1].cpu().numpy().tolist()
                running_corrects += torch.sum(preds == labs.data)
                for i in img_path:
                	incorrect_path.append(i)

            epoch_acc = running_corrects.double() / val_size
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())


                #if ep == 0:
                #    for idx, k in enumerate(img_path):
                #    	incorrect_path.append(str(img_path[idx]) + ' ' + str(labels[idx]) + ' ' + str(predictions[idx]))


            test_metrics.append(np.mean(batch_metrics_test))

            label_array = np.asarray(label_list, dtype=np.int).ravel()
            output_array = np.asarray(output_list, dtype=np.int).ravel()
            img_array = np.asarray(incorrect_path).ravel()

            calculate_metrics(output_array, label_array)

            if ep == epochs-1:

                for i in range(len(img_array)):
                    if criterion_name == "fl":
                        print(str(img_array[i]) + ' ' + str(label_array[i]) + ' ' + str(output_array[i]), file=open(model_name+'_'+'gamma_'+str(gamma)+'bs12.txt', "a"))
                    else:
                        print(str(img_array[i]) + ' ' + str(label_array[i]) + ' ' + str(output_array[i]), file=open(model_name+'_'+'CE_'+'bs12.txt', "a"))

    # Save stuff
    model.load_state_dict(best_model)
    labels = np.asarray(labels, dtype=np.int).ravel()
    predictions = np.asarray(predictions, dtype=np.int).ravel()
    if criterion_name == "fl":
        torch.save(model.state_dict(), model_name+'_'+'gamma_'+str(gamma) + "bs12.pt")
    else:
        torch.save(model.state_dict(), model_name+'_'+'CE_'+ "bs12.pt")
    # model.load_state_dict(torch.load('Densenet169.pt'))

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

    plt.savefig("charts_balanced3" + ".png")



#################### FAZ AS COISAS #####################
epochs = 10
num_classes = 2
lr = 1e-5
num_folds = 3
mod = 'resnet152_bin'
#mod = 'vit'
vit = False
if mod == 'vit':
    vit = True
gpu = 0
batch_size = 12

#for i in range (num_folds):
#    print('FOLD : ', i)
#    root = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/'
#    csv_train = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/fold/train{}.csv'.format(i+1)
#    csv_test = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/fold/test{}.csv'.format(i+1)
#
#    model = get_classification_model(num_classes, mod)
#    data_loader, dataset, pesos = set_dataset()
#    go(model, data_loader, dataset, lr, epochs, pesos, num_classes, gpu)


root = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/'
csv_train = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/fold/train2.csv'
csv_test = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/fold/test2.csv'
#root = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/'
#csv_train = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/fold/train3.csv'
#csv_test = '/mnt/DADOS_PARIS_1/ester/icm/fachadas/preprocessed2/fold/test3.csv'
#root = '/mnt/DADOS_PARIS_1/matheusp/csv_segunda_leva/'
#csv_train = '/mnt/DADOS_PARIS_1/joaopedro/fachadas2/folds/train1.csv'
#csv_test = '/mnt/DADOS_PARIS_1/joaopedro/fachadas2/folds/test1.csv'
model = get_classification_model(num_classes, mod)
data_loader, dataset, pesos = set_dataset(batch_size, vit)
go(model, data_loader, dataset, lr, epochs, pesos, num_classes, gpu, mod)