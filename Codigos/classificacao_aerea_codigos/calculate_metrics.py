from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, cohen_kappa_score, f1_score
import sys
import numpy as np
from joblib import load
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--img_type', type=str, required=True)
# args = parser.parse_args()

# Allow print matrix without truncation
np.set_printoptions(threshold=sys.maxsize)

def calculate_metrics(preds, labels):
    cm = confusion_matrix(np.asarray(labels), np.asarray(preds))
    b_acc = balanced_accuracy_score(np.asarray(labels), np.asarray(preds))
    acc = accuracy_score(np.asarray(labels), np.asarray(preds))
    kappa = cohen_kappa_score(np.asarray(labels), np.asarray(preds))
    f1 = f1_score(np.asarray(labels), np.asarray(preds), average = 'weighted')  
        
    print ("\nAccuracy: " + str(acc))
    print ("Balanced_Accuracy: " + str(b_acc))
    print ("Kappa: " + str(kappa))
    print ("F1: " + str(f1))
    print (cm)


def new_metrics(preds, labels):

  labels = [0 if i==1 else i for i in labels]
  labels = [1 if i==2 else i for i in labels]
  labels = [2 if i==3 else i for i in labels]
  labels = [2 if i==4 else i for i in labels]

  preds = [0 if i==1 else i for i in preds]
  preds = [1 if i==2 else i for i in preds]
  preds = [2 if i==3 else i for i in preds]
  preds = [2 if i==4 else i for i in preds]

  calculate_metrics(preds, labels)



