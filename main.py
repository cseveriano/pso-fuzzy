import PSO
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

def sinc(xx):
    return math.sin(xx)/xx

def initProblemSinc():
    x = np.arange(0.01, 2 * math.pi, 0.01)
    y = [sinc(xx) for xx in x]

    return [x,y]

def normalize(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.StandardScaler()
    scaler = min_max_scaler.fit(x)
    x_scaled = scaler.transform(x)
    df_norm = pd.DataFrame(x_scaled)
    return df_norm, scaler

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

df = pd.read_csv('../Data/train.csv')
df_test = pd.read_csv('../Data/test.csv')

x = df[['OverallQual','GrLivArea','GarageCars']]
x, scalerx = normalize(x)
x.columns = ['OverallQual','GrLivArea','GarageCars']

y = df['SalePrice']
y, scalery = normalize(y)

X_test = df_test[['OverallQual','GrLivArea','GarageCars']]
X_test.GarageCars[1116] = 0 #ajustando valor nulo
X_test, scalerx_test = normalize(X_test)
X_test.columns = ['OverallQual','GrLivArea','GarageCars']


#[x,y] = initProblemSinc()
#[x,y] = initProblemHousePrice()

# Teste para submissao
X_train = x
y_train = y

# Teste com base de treinamento
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

nrules = 8

pso = PSO.PSO(X_train,nrules, num_particles=20,maxiter=100)
model = pso.train(X_train,y_train)


y_est = model.yEstimated(X_test)
#print("Erro Teste: ",error,"\n")


#y_denorm = scalery.inverse_transform(y_test)
y_est_denorm = scalery.inverse_transform(y_est)
#y_est_denorm = y_est_denorm.astype(np.int64)
#rmse = sqrt(mean_squared_error(y_denorm, y_est_denorm))
#rmse = sqrt(mean_squared_error(y_test, y_est))
#print("Erro Teste: ",rmse,"\n")

my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': y_est_denorm})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
## Dividir base train com 70-30 usando CV
