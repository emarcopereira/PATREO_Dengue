import pandas as pd
import numpy as np

dados = pd.read_csv('ciclo_ICM.csv')



counts = dados['label'].value_counts()
print(counts)
weights = []
for i in range(2):
    weights.append(1-counts[i+1]/3675)

w_norm = [w / sum(weights) for w in weights]

print(weights)
print(w_norm)
