#@Author Luis Miguel Aguilar González

#Puede que sea necesaria la instalación de 
#alguna librería para su correcta ejecución
#como puede ser Scipy usando el comando de
#pip install scipy
#Al ejecutarlo será necesario seleccionar uno de los
#preprocesados introduciendo 1,2 ó 3 y pulsando enter

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cmath
import pylab as pl
import copy
from matplotlib.axes import  Axes
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import impute
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from imblearn.metrics import geometric_mean_score
from scipy.stats import gmean

def norm(data):
    """Normaliza una serie de datos"""
    return (data - data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))

fichero = "data/mamografias.csv"
data = pd.read_csv(fichero,na_values=['?']) 

print("Seleccione preprocesado de datos:\n\t 1.- Eliminar valores nulos\n\t 2.- Reemplazo Simple\n\t 3.- Reemplazo Knn")
preprocesado = int(input())

if preprocesado == 1:
    data = data.dropna()
    #create the Labelencoder object
    le = preprocessing.LabelEncoder()
    #convert the categorical columns into numeric
    data["Severity"] = le.fit_transform(data["Severity"])
    data["Shape"] = le.fit_transform(data["Shape"])

if preprocesado == 2:
    imputer = impute.SimpleImputer(strategy="most_frequent")
    birads = imputer.fit_transform([data['BI-RADS'].values])
    Age = imputer.fit_transform([data['Age'].values])
    Shape = imputer.fit_transform([data['Shape'].values])
    Margin = imputer.fit_transform([data['Margin'].values])
    Density = imputer.fit_transform([data['Density'].values])
    Severity = imputer.fit_transform([data['Severity'].values])
    data['BI-RADS'].update(pd.Series(birads[0]))
    data['Age'].update(pd.Series(Age[0]))
    data['Shape'].update(pd.Series(Shape[0]))
    data['Margin'].update(pd.Series(Margin[0]))
    data['Density'].update(pd.Series(Density[0]))
    data['Severity'].update(pd.Series(Severity[0]))
    #create the Labelencoder object
    le = preprocessing.LabelEncoder()
    #convert the categorical columns into numeric
    data["Severity"] = le.fit_transform(data["Severity"])
    data["Shape"] = le.fit_transform(data["Shape"])

if preprocesado == 3:
    #create the Labelencoder object
    le = preprocessing.LabelEncoder()
    #convert the categorical columns into numeric
    data["Severity"] = le.fit_transform(data["Severity"])
    data["Shape"] = le.fit_transform(data["Shape"])
    imputer = impute.KNNImputer()
    birads = imputer.fit_transform([data['BI-RADS'].values])
    Age = imputer.fit_transform([data['Age'].values])
    Shape = imputer.fit_transform([data['Shape'].values])
    Margin = imputer.fit_transform([data['Margin'].values])
    Density = imputer.fit_transform([data['Density'].values])
    Severity = imputer.fit_transform([data['Severity'].values])
    data['BI-RADS'].update(pd.Series(birads[0]))
    data['Age'].update(pd.Series(Age[0]))
    data['Shape'].update(pd.Series(Shape[0]))
    data['Margin'].update(pd.Series(Margin[0]))
    data['Density'].update(pd.Series(Density[0]))
    data['Severity'].update(pd.Series(Severity[0]))

print(data.shape)


# Mostrar datos en función de los distintos parametros
f, axes = plt.subplots(2, 2, figsize=(9, 9))
sns.set(style="whitegrid", color_codes = True)
sns.countplot('BI-RADS',data=data, hue = 'Severity',ax=axes[0,0])
sns.countplot('Margin',data=data, hue = 'Severity', ax=axes[0,1])
sns.countplot('Shape',data=data, hue = 'Severity', ax=axes[1,0])
sns.countplot('Density',data=data, hue = 'Severity', ax=axes[1,1])
plt.show() 

sns.set(style="whitegrid", color_codes = True)
sns.countplot('Severity',data=data)
plt.show()

plt.figure(figsize=(20,20))
sns.heatmap(data.corr(), annot=True, fmt='.0%')
plt.show()


sns.pairplot(data,hue='Severity')
plt.savefig('pairplotreemplazoknn.png')
plt.show()

cols = [col for col in data.columns if col not in ['Severity']]

eval_data = data[cols] 
target = data['Severity']

data_train, data_test, target_train, target_test = train_test_split(eval_data,target, test_size = 0.30, random_state = 10)


#NAIVE BAYES
gnb = GaussianNB()

model1 = gnb.fit(data_train, target_train)

pred1 = model1.predict(data_test)

print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred1, normalize = True))

#Multi-layer Perceptron 

mlp = MLPClassifier(random_state=1,max_iter=500)

model2 = mlp.fit(data_train, target_train)

pred2 = model2.predict(data_test)

print("MLP accuracy : ",accuracy_score(target_test, pred2, normalize = True))

#K-nn algorithm 

k_range = range(1,560)
scores= {}
scores_list = []
mayor = 0


norm_data = copy.deepcopy(data)
norm_data_f = norm(norm_data)
norm_eval_data = norm_data_f[cols] 
norm_target = norm_data_f['Severity']

X_train, X_test, y_train, y_test = train_test_split(norm_eval_data, norm_target, test_size=0.30)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    pred=knn.predict(X_test)
    scores[k] = accuracy_score(y_test, pred)
    if mayor < scores[k]:
        mayor = scores[k]
        k_fin = k    
    scores_list.append(accuracy_score(y_test, pred))

knn = KNeighborsClassifier(n_neighbors=k_fin)
knn.fit(X_train, y_train)

plt.plot(k_range,scores_list)
plt.xlabel('Value of k for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

print("K-nn accuracy for k =", k_fin,":", scores_list[k_fin])

#DecisionTreeClassifier optimized CART algorithm
print("\nDecision tree, CART algorithm:\n")
 
fn=['BI-RADS','Age','Shape','Margin','Density','Severity']
cn=['benigno','maligno']

tree_model = tree.DecisionTreeClassifier()
model_tree = tree_model.fit(data_train,target_train)
max_depth = model_tree.tree_.max_depth

depth_range = range(1,max_depth)

mayor = 0
scores2 = {}
scores_list2 = []

# Testearemos la profundidad de 1 a cantidad de atributos +1
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(max_depth = depth)
    model_tree = tree_model.fit(data_train,target_train) 
    pred4 = model_tree.predict(data_test)
    scores2[depth] = accuracy_score(target_test, pred4, normalize = True)
    if mayor < scores2[depth]:
        mayor = scores2[depth]
        depth_fin = depth

    scores_list2.append(accuracy_score(target_test, pred4, normalize = True))

plt.plot(depth_range, scores_list2)
plt.xlabel('Value of depth for CART tree')
plt.ylabel('Testing Accuracy')
plt.show()

print("Depth:",depth_fin,"with max accuracy:", mayor)

tree_model = tree.DecisionTreeClassifier(max_depth = depth_fin)
model_tree = tree_model.fit(data_train,target_train)


fig1 = plt.figure(figsize=(25,25))
tree.plot_tree(model_tree, max_depth = depth_fin,
                           impurity = True,
                           feature_names = fn,
                           class_names = cn,
                           rounded = True,
                           filled= True )

fig1.savefig('ArbolCART-Aguilar-Gonzalez-LuisMiguel.png')

#RandomForestClassifier

#number of levels in tree
max_depth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
grid_param = {'max_depth': max_depth}
RFR = RandomForestRegressor(random_state=1)
RFR_random = RandomizedSearchCV(estimator= RFR, 
                                param_distributions= grid_param,
                                n_iter=500, 
                                cv= 5, 
                                verbose=2, 
                                random_state=42, 
                                n_jobs= -1)
RFR_random.fit(data_train,target_train)

reg = RandomForestClassifier(max_depth=RFR_random.best_params_["max_depth"])

reg = reg.fit(data_train,target_train)

pred5 =  reg.predict(data_test)
print ("Depth randomForest: ", RFR_random.best_params_["max_depth"])
print("Random Forest accuracy", accuracy_score(target_test, pred5, normalize = True))

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(reg.estimators_[0], max_depth = RFR_random.best_params_["max_depth"],
                      impurity = True,
                      feature_names = fn,
                      class_names = cn,
                      rounded = True,
                      filled= True )
fig.savefig('ArbolRandomForest-Aguilar-Gonzalez-LuisMiguel.png')


#----------------------------------------------------------------------------------------------------------------------

#Cross validation Naive Bayes
cv_scores = cross_val_score(model1, eval_data, target, cv=5)

#print each cv score (accuracy) and average them
print(cv_scores)
print('Naive bayes cv_scores mean:{}'.format(np.mean(cv_scores)))

#Cross validation k-nn algorithm
cv_scores = cross_val_score(knn, eval_data, target, cv=5)

#print each cv score (accuracy) and average them
print(cv_scores)
print('K-nn algorithm cv_scores mean:{}'.format(np.mean(cv_scores)))

#Cross validation Multi-layer Perceptron
cv_scores = cross_val_score(model2, eval_data, target, cv=5)

#print each cv score (accuracy) and average them
print(cv_scores)
print('Multi-layer Perceptron cv_scores mean:{}'.format(np.mean(cv_scores)))

#Cross validation CART Algorithm
cv_scores = cross_val_score(tree_model, eval_data, target, cv=5)

#print each cv score (accuracy) and average them
print(cv_scores)
print('CART algorithm cv_scores mean:{}'.format(np.mean(cv_scores)))

#Cross validation Random forest
cv_scores = cross_val_score(reg, eval_data, target, cv=5)

#print each cv score (accuracy) and average them
print(cv_scores)
print('Random Forest cv_scores mean:{}'.format(np.mean(cv_scores)))


#----------------------------------------------------------------------------------------------------------------------
#Confusion Matrix
print("\n")

#Naive Bayes
print("Confusion matrix Naive Bayes:\n")
predNaive = model1.predict(data_test)
print(confusion_matrix(predNaive, target_test))
plot_confusion_matrix(model1,data_test,target_test)
plt.show()
print("\n")

#Multi-layer Perceptron
print("Confusion matrix Multi-layer Perceptron:\n")
predMulti = model2.predict(data_test)
print(confusion_matrix(predMulti, target_test))
plot_confusion_matrix(model2,data_test,target_test)
plt.show()
print("\n")

#k-nn algorithm
print("Confusion matrix k-nn algorithm:\n")
predknn = knn.predict(X_test)
print(confusion_matrix(predknn, y_test))
plot_confusion_matrix(knn,X_test,y_test)
plt.show()
print("\n")

#CART Algorithm
print("Confusion matrix CART Algorithm:\n")
predCart = tree_model.predict(data_test)
print(confusion_matrix(predCart,target_test))
plot_confusion_matrix(tree_model,data_test,target_test)
plt.show()
print("\n")

#Random forest
print("Confusion matrix Random forest:\n")
predRandom = reg.predict(data_test)
print(confusion_matrix(predRandom,target_test))
plot_confusion_matrix(reg,data_test,target_test)
plt.show()
print("\n")

#----------------------------------------------------------------------------------------------------------------------
#ROC and AUC

#Naive Bayes

fig, ax = plt.subplots(2,3, sharex=True, sharey=True)

fig.suptitle('Reemplazo knn') #Insertar subtitulos del preprocesado
kf = KFold(n_splits=5);

for train_index, test_index in kf.split(eval_data, target):
    
    probNaives = model1.predict_proba(eval_data.iloc[test_index])
    fpr, tpr, thresholds = roc_curve(target.iloc[test_index],probNaives[:,1])
    roc_auc = auc(fpr, tpr)

    ax[0,0].plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
    ax[0,0].set_title('Receiver operating characteristic Naive Bayes')
    ax[0,0].plot([0, 1], [0, 1], 'k--')
    ax[0,0].legend(loc="lower right")


#Multi-layer Perceptron

kf = KFold(n_splits=5);

for train_index, test_index in kf.split(eval_data, target):
    probMultiLayer = model2.predict_proba(eval_data.iloc[test_index])
    fpr, tpr, thresholds = roc_curve(target.iloc[test_index],probMultiLayer[:,1])
    roc_auc = auc(fpr, tpr)

    ax[0,1].plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax[0,1].set_title('Receiver operating characteristic Multi-layer Perceptron')
    ax[0,1].plot([0, 1], [0, 1], 'k--')
    ax[0,1].legend(loc="lower right")


#k-nn Algorithm

kf = KFold(n_splits=5);

for train_index, test_index in kf.split(norm_eval_data, target):
    probknn = knn.predict_proba(norm_eval_data.iloc[test_index])
    fpr, tpr, thresholds = roc_curve(norm_target.iloc[test_index],probknn[:,1])
    roc_auc = auc(fpr, tpr)
    if roc_auc > 0:
        ax[0,2].plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        ax[0,2].set_title('Receiver operating characteristic k-nn algorithm')
        ax[0,2].plot([0, 1], [0, 1], 'k--')
        ax[0,2].legend(loc="lower right")

# plt.show()

#CART Algorithm

kf = KFold(n_splits=5);

for train_index, test_index in kf.split(eval_data, target):
    probCART = tree_model.predict_proba(eval_data.iloc[test_index])
    fpr, tpr, thresholds = roc_curve(target.iloc[test_index],probCART[:,1])
    roc_auc = auc(fpr, tpr)

    ax[1,0].plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax[1,0].set_title('Receiver operating characteristic CART algorithm')
    ax[1,0].plot([0, 1], [0, 1], 'k--')
    ax[1,0].legend(loc="lower right")

#Random forest

kf = KFold(n_splits=5);

for train_index, test_index in kf.split(eval_data, target):
    probRandom = reg.predict_proba(eval_data.iloc[test_index])
    fpr, tpr, thresholds = roc_curve(target.iloc[test_index],probRandom[:,1])
    roc_auc = auc(fpr, tpr)


    ax[1,1].plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax[1,1].set_title('Receiver operating characteristic Random forest')
    ax[1,1].plot([0, 1], [0, 1], 'k--')
    ax[1,1].legend(loc="lower right")

for a in ax.flat:
    a.set(xlabel='False Positive Rate', ylabel= 'True Positive Rate')

for a in ax.flat:
    a.label_outer()

ax[1,2].set_visible(False)
plt.show()


#--------------------------------------------------------------------------------
#Accuracy bar graph

precision_list = []

precision_list.append(accuracy_score(target_test, predNaive, normalize = True))
precision_list.append(accuracy_score(target_test, predMulti, normalize = True))
precision_list.append(accuracy_score(y_test, predknn, normalize = True))
precision_list.append( accuracy_score(target_test, predCart, normalize = True))
precision_list.append(accuracy_score(target_test, predRandom, normalize = True))

plt.bar(['Naive','Multi-Layer', 'Knn', 'CART', 'Random Forest'], precision_list)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()



