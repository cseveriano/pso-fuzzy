import PSO
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

nrules = 5

pso = PSO.PSO(X_train,nrules, num_particles=10,maxiter=100)
model = pso.train(X_train,y_train)

error = model.test(X_test,y_test)
print("Erro Teste: ",error,"\n")


## Dividir base train com 70-30 usando CV
