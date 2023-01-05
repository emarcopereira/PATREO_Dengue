import numpy as np
import pandas as pd

threshold = 2800

#fold 1
csv = pd.read_csv('/mnt/DADOS_PARIS_1/joaopedro/fachadas2/ciclo_ICM.csv')
train1 = csv[0:threshold]
test1 = csv[threshold+1:]
train1.to_csv('/mnt/DADOS_PARIS_1/joaopedro/fachadas2/folds/train1.csv')
test1.to_csv('/mnt/DADOS_PARIS_1/joaopedro/fachadas2/folds/test1.csv')

#fold 2
d1 = csv.sample(frac=1)
train2 = d1[0:threshold]
test2 = d1[threshold+1:]
train2.to_csv('/mnt/DADOS_PARIS_1/joaopedro/fachadas2/folds/train2.csv')
test2.to_csv('/mnt/DADOS_PARIS_1/joaopedro/fachadas2/folds/test2.csv')

#fold 3
d2 = d1.sample(frac=1)
train3 = d2[0:threshold]
test3 = d2[threshold+1:]
train3.to_csv('/mnt/DADOS_PARIS_1/joaopedro/fachadas2/folds/train3.csv')
test3.to_csv('/mnt/DADOS_PARIS_1/joaopedro/fachadas2/folds/test3.csv')
