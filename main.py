import PSO
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing

def sinc(xx):
    return math.sin(xx)/xx

def initProblemSinc():
    x = np.arange(0.01, 2 * math.pi, 0.01)
    y = [sinc(xx) for xx in x]

    return [x,y]

def normalize(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_norm = pd.DataFrame(x_scaled)
    df_norm.columns = df.columns
    return df_norm

def initProblemHousePrice():
    df = pd.read_csv('../Data/train.csv')

    df_train = df[['OverallQual','GrLivArea','GarageCars','SalePrice']]
    df_train = normalize(df_train)

    x = df_train[['OverallQual','GrLivArea','GarageCars']]
    y = df_train['SalePrice']

    return [x,y]



## separar colunas mais correlacionadas (3)
## definir numero de parametros com base no numero de atributos
## enviar para o PSO


#[x,y] = initProblemSinc()
[x,y] = initProblemHousePrice()

nrules = 5

PSO.PSO(x,y,nrules, num_particles=10,maxiter=100)


## Dividir base train com 70-30 usando CV
