from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, cohen_kappa_score, f1_score
import sys
import numpy as np
from joblib import load
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_type', type=str, required=True)
args = parser.parse_args()

# Allow print matrix without truncation
np.set_printoptions(threshold=sys.maxsize)

def calculate_metrics(preds, labels, file = None):
    cm = confusion_matrix(np.asarray(labels), np.asarray(preds))
    b_acc = balanced_accuracy_score(np.asarray(labels), np.asarray(preds))
    acc = accuracy_score(np.asarray(labels), np.asarray(preds))
    kappa = cohen_kappa_score(np.asarray(labels), np.asarray(preds))
    f1 = f1_score(np.asarray(labels), np.asarray(preds), average = 'weighted')  
    if file is not None:
        file.write("Accuracy: " + str(acc) + "\n")
        file.write("Balanced_Accuracy: " + str(b_acc) + "\n")
        file.write("Kappa: " + str(kappa) + "\n")
        file.write("F1: " + str(f1) + "\n")
        #file.write("Confusion Matrix: " + "\n\n\n\n")
        file.write("Labels\n{}\n".format(np.asarray(labels)))
        file.write("Predictions\n{}".format(np.asarray(preds)))
        
    else:
      print ("\nAccuracy: " + str(acc))
      print ("Balanced_Accuracy: " + str(b_acc))
      print ("Kappa: " + str(kappa))
      print ("F1: " + str(f1))
      print (cm)


if(args.img_type == "aerial"):
  label = load("test_aerial/labels_balanced_aerial.joblib")
  pred = load("test_aerial/predictions_balanced_aerial.joblib")
else:
  label = load("test_street/labels_balanced_street.joblib")
  pred = load("test_street/predictions_balanced_street.joblib")

for i, v in enumerate(label):
  if v < 2:
    label[i] = 0
  elif v < 4:
    label[i] = 1
  else:
    label[i] = 2


for i, v in enumerate(pred):
  if v < 2:
    pred[i] = 0
  elif v < 4:
    pred[i] = 1
  else:
    pred[i] = 2

calculate_metrics(pred, label)