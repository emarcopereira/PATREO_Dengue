import pandas as pd
import numpy as np

dados1 = pd.read_csv('conferir_qgis.csv')


dados1 = dados1[['foto', 'fachada']]

#dados1['foto'] = dados1['foto'].str.split('/')
#dados1['foto'] = dados1['foto'].str[2]


dados1['fachada'] = dados1['fachada'].astype(str)
dados1['fachada'] = dados1['fachada'].replace('2', '1')
dados1['fachada'] = dados1['fachada'].replace('3', '2')
dados1['fachada'] = dados1['fachada'].replace('4', '3')
dados1['fachada'] = dados1['fachada'].replace('5', '3')


# 1 -> 1
# 2 -> 1
# 3 -> 2
# 4,5 -> 3

dados1 = dados1.rename(columns={"foto": "arquivo", "fachada": "label"})


dados1.to_csv('ciclo_ICM.csv')
