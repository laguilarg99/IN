#@Author Luis Miguel Aguilar González


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cmath
import pylab as pl
import copy
import imblearn
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from imblearn.datasets import make_imbalance
from sklearn.datasets import make_moons
from collections import Counter
from sklearn import tree
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn import impute
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from collections import Counter
from xgboost import XGBClassifier


def norm(data):
    """Normaliza una serie de datos"""
    return (data - data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))

fichero_train = "data/train.csv"
fichero_test = "data/test.csv"


namestrain = ['Nombre','Ciudad', 'Año', 'Kilometros', 'Combustible', 'Tipo_marchas', 'Potencia','Mano', 'Asientos', 'Precio_cat']
names = ['Nombre','Ciudad', 'Año', 'Kilometros', 'Combustible', 'Tipo_marchas' ,'Potencia', 'Mano', 'Asientos']

#Data train
data_train = pd.read_csv(fichero_train, na_values=['?'])
data_train = data_train[namestrain].copy()

# print(data_train.shape)

data_train.drop_duplicates(inplace=True)

#Data test
data_test = pd.read_csv(fichero_test, na_values=['?'])
ids = data_test['id']
del data_test['id']

# print(data_test.shape)
data_test = data_test[names].copy()


#Elimina Str de los valores de las columnas
data_test['Nombre'] = data_test['Nombre'].str.split(' ').str[0]
data_test['Potencia'] = data_test['Potencia'].str.replace(r'\D', '')

####---------------------------------------------------------------------
# #Permite transformar el consumo y el motor a un valor númerico
# data_test['Consumo'] = data_test['Consumo'].str.replace(r'\D', '')
# data_test['Motor_CC'] = data_test['Motor_CC'].str.replace(r'\D', '')


#Elimina Str de los valores de las columnas
data_train['Nombre'] = data_train['Nombre'].str.split(' ').str[0]
data_train['Potencia'] = data_train['Potencia'].str.replace(r'\D', '')

####---------------------------------------------------------------------
# #Permite transformar el consumo y el motor a un valor númerico
# data_train['Consumo'] = data_train['Consumo'].str.replace(r'\D', '')
# data_train['Motor_CC'] = data_train['Motor_CC'].str.replace(r'\D', '')

#Eliminar valores nulos
# print(data_train.isnull().sum())
data_train['Potencia'] = pd.to_numeric(data_train['Potencia'])

# Reemplaza con la media de la columna
# data_train['Kilometros'] = data_train['Kilometros'].fillna(data_train['Kilometros'].mean()) 

data_train['Potencia'] = data_train['Potencia'].fillna(data_train['Potencia'].value_counts().index[0])

#Reemplaza con el valor que más se repite
# data_train['Año'] = data_train['Año'].fillna(data_train['Año'].value_counts().index[0]) 

data_train['Combustible'] = data_train['Combustible'].fillna(data_train['Combustible'].value_counts().index[0])
# data_train['Nombre'] = data_train['Nombre'].fillna(data_train['Nombre'].value_counts().index[0])
# data_train['Asientos'] = data_train['Asientos'].fillna(data_train['Asientos'].value_counts().index[0])
# data_train['Descuento'] = data_train['Descuento'].fillna(0)
data_train = data_train.fillna(method='ffill')


#Valores nulos data_test
# data_test['Descuento'] = data_test['Descuento'].fillna(0)

#Transforma los datos a str
data_train['Combustible'] = data_train['Combustible'].values.astype(str)
data_train['Tipo_marchas'] = data_train['Tipo_marchas'].values.astype(str)
data_train['Mano'] = data_train['Mano'].values.astype(str)

#data test
data_test['Mano'] = data_test['Mano'].values.astype(str)

#Convierte aquellas columnas cualitativas en cuantitativas
labelCombustible = preprocessing.LabelEncoder().fit(pd.read_csv("data/combustible.csv").Combustible)
labelTipoMarcha = preprocessing.LabelEncoder().fit(pd.read_csv("data/tipo_marchas.csv").Tipo_marchas)
labelMano = preprocessing.LabelEncoder().fit(pd.read_csv("data/mano.csv").Mano)
labelNombre = preprocessing.LabelEncoder().fit(pd.read_csv("data/nombre.csv").Nombre.str.split(' ').str[0])
# data_train['Combustible'] = labelCombustible.transform(data_train['Combustible'])
data_train['Tipo_marchas'] = labelTipoMarcha.transform(data_train['Tipo_marchas'])
data_train['Mano'] = labelMano.transform(data_train['Mano'])
data_train['Nombre'] = labelNombre.transform(data_train['Nombre'])


#Data test
# data_test['Combustible'] = labelCombustible.transform(data_test['Combustible'])
data_test['Tipo_marchas'] = labelTipoMarcha.transform(data_test['Tipo_marchas'])
data_test['Mano'] = labelMano.transform(data_test['Mano'])
data_test['Nombre'] = labelNombre.transform(data_test['Nombre'])


#Transforma los datos a int
data_train['Año'] = data_train['Año'].values.astype(int)
data_train['Kilometros'] = data_train['Kilometros'].values.astype(int)
data_train['Potencia'] = data_train['Potencia'].values.astype(int)
# data_train['Consumo'] = data_train['Consumo'].values.astype(int)
# data_train['Motor_CC'] = data_train['Motor_CC'].values.astype(int)

#data_test
data_test['Potencia'] = data_test['Potencia'].values.astype(int)
# data_test['Consumo'] = data_test['Consumo'].values.astype(int)
# data_test['Motor_CC'] = data_test['Motor_CC'].values.astype(int)

#Reduce los valores del conjunto de datos, estan todas las restricciones que he probado
data_train = data_train[
        (data_train['Combustible'] != 'Electric')
        & (data_train['Kilometros'] <= 600000)
        # & (data_train['Combustible'] != 'LPG')
        # & (data_train['Descuento'] < 55.0)
        # & (data_train['Asientos'] > 0.0)
        # & (data_train['Potencia'] >= 45.0)
        # & (data_train['Potencia'] <= 1000.0)
        ]
        

#Se encarga de transformar los atributos ciudad, combustible y marchas a atributos binarios

var = 'Ciudad'
# print(data_train['Ciudad'].value_counts())
Location = data_train[[var]]
Location = pd.get_dummies(Location,drop_first=True)
# print(Location.head())

#Test
Location_test = data_test[[var]]
Location_test = pd.get_dummies(Location_test,drop_first=True)

var = 'Combustible'
# print(data_train['Combustible'].value_counts())
Fuel_t = data_train[[var]]
Fuel_t = pd.get_dummies(Fuel_t, drop_first=True)
# print(Fuel_t.head())

#Test
Fuel_t_test = data_test[[var]]
Fuel_t_test = pd.get_dummies(Fuel_t_test, drop_first=True)

var = 'Tipo_marchas'
Transmission = data_train[[var]]
Transmission = pd.get_dummies(Transmission, drop_first=True)
Transmission.columns=['Transmission']
# print(Transmission.head())

#Test
Transmission_test = data_test[[var]]
Transmission_test = pd.get_dummies(Transmission_test, drop_first=True)
Transmission_test.columns=['Transmission']

precio = data_train['Precio_cat']
del data_train['Precio_cat']
data_train = pd.concat([data_train, Location, Fuel_t, Transmission, precio], axis=1)
data_train.drop(['Ciudad', 'Combustible', 'Tipo_marchas'],axis=1,inplace=True)


#Test
data_test = pd.concat([data_test, Location_test, Fuel_t_test, Transmission_test], axis=1)
data_test.drop(['Ciudad', 'Combustible', 'Tipo_marchas'],axis=1,inplace=True)


# Con este fragmento de código pretendía calcular los cuartiles para establecer 
# un rango de valores automático pero reducía la muestra drásticamente 

# print(data_train.shape)
# print(data_test.dtypes)
# print(data_train.dtypes)

# Q1 = data_train.quantile(0.25)
# Q2 = data_train.quantile(0.5)
# Q3 = data_train.quantile(0.75)

# IQR = Q3-Q1
# Min = Q1-(1.5*IQR)
# Max = Q3+(1.5*IQR)

# print("IQR : ",IQR)
# print("")
# print("Min : ",Min)
# print("")
# print("Q1 : ",Q1)
# print("")
# print("Q2 : ",Q2)
# print("")
# print("Q3 : ",Q3)
# print("")
# print("Max : ",Max)

# columnas = data_train.loc[:, data_train.columns != 'Precio_cat']
# data_train = data_train[~((data_train[columnas.columns] < Min) | (data_train[columnas.columns] > Max)).any(axis=1)]

# print(data_train.columns)
# data_train.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=True, fontsize=1, figsize=(15,12))
# plt.show()

# print(data_train.isnull().sum())

#SMOTE Synthetic Minority Oversampling Technique
X = data_train.loc[:, data_train.columns != 'Precio_cat']
y = data_train.Precio_cat
oversample = SMOTE(sampling_strategy='not majority', random_state=5)
randomoversample = RandomOverSampler(sampling_strategy='minority')
X_res, y_res = oversample.fit_resample(X, y)

data_train = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)
# print(data_train.columns)

# Muestra los valores de los distintos atributos del conjunto de test
# for i, c in enumerate(data_test.columns):
#     v = data_test[c].unique()
    
#     g = data_test.groupby(by=c)[c].count().sort_values(ascending=False)
#     r = range(min(len(v), 5))

#     print( g.head())
#     plt.figure(figsize=(5,3))
#     plt.bar(r, g.head()) 
#     #plt.xticks(r, v)
#     plt.xticks(r, g.index)
#     plt.show()

# print(data_test.shape)
# data_train.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=True, fontsize=1, figsize=(15,12))
# plt.show()

# print(data_train.shape)
# print(data_test.isnull().sum())

# data_target = data_train['Precio_cat'].copy()
# data_train = data_train[names].copy()

# data_train | data_target
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.30, random_state = 5)

max_depth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
grid_param = {'max_depth': max_depth}
RFR = RandomForestRegressor(random_state=1)
RFR_random = RandomizedSearchCV(estimator= RFR, 
                                param_distributions= grid_param,
                                n_iter=500, 
                                cv= 5, 
                                verbose=2, 
                                random_state=42, 
                                n_jobs= -1)
RFR_random.fit(x_train,y_train)

reg = RandomForestClassifier(max_depth=RFR_random.best_params_["max_depth"])

reg = reg.fit(x_train,y_train)

pred5 =  reg.predict(x_test)

print ("Depth randomForest: ", RFR_random.best_params_["max_depth"])
print("Random Forest accuracy", accuracy_score(y_test, pred5, normalize = True))

#Cross validation CART Algorithm
cv_scores = cross_val_score(reg, x_train, y_train, cv=5)

# print each cv score (accuracy) and average them
print(cv_scores)
print('Random Forest algorithm cv_scores mean:{}'.format(np.mean(cv_scores)))


# Calcula la mejor semilla para el algoritmo de XGBClassifier
# mejor_score = 0
# mejor_i = 0

# for i in range(0,20):
#     test_size = 0.30

#     X_train, X_val, y_train, y_val = train_test_split(X_res, y_res,  
#             test_size=test_size, random_state=i)

#     xgb_clf = XGBClassifier()
#     xgb_clf.fit(X_train, y_train)

#     predict = xgb_clf.predict(X_val)
#     score = xgb_clf.score(X_val, y_val)
#     if mejor_score < score:
#         mejor_score = score
#         mejor_i = i

#     print(i," ", score)


# #GRADIENT BOOSTING junto con la lista de valores para probar distintos learning_rate
# # lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

# # for learning_rate in lr_list:
# #       gb_clf = GradientBoostingClassifier(n_estimators=120, learning_rate=learning_rate, max_features=5, max_depth=5, random_state=0)
# #       gb_clf.fit(X_train, y_train)
    
# #       print("Learning rate: ", learning_rate)
# #       print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
# #       print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))


# #XGBOOST
# state = mejor_i
# test_size = 0.30

# X_train, X_val, y_train, y_val = train_test_split(X_res, y_res,  
#     test_size=test_size, random_state=state)

# xgb_clf = XGBClassifier()
# xgb_clf.fit(X_train, y_train)


predict = reg.predict(x_test)

print(classification_report(y_test, predict, target_names=['1','2','3','4','5']))

print(data_test.isnull().sum())

predict = reg.predict(data_test)

df_result = pd.DataFrame({'id': ids, 'Precio_cat': predict})
df_result.to_csv("mis_resultados.csv", index=False)
# score = xgb_clf.score(X_val, y_val)
print(data_train.shape)
# print(score)