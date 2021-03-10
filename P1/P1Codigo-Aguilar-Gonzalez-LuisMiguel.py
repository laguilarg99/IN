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
from sklearn import impute
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from imblearn.metrics import geometric_mean_score
from scipy.stats import gmean




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

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data_train,target_train)
    pred=knn.predict(data_test)
    scores[k] = accuracy_score(target_test, pred)
    if mayor < scores[k]:
        mayor = scores[k]
        k_fin = k    
    scores_list.append(accuracy_score(target_test, pred))

knn = KNeighborsClassifier(n_neighbors=k_fin)
knn.fit(data_train,target_train)

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
predknn = knn.predict(data_test)
print(confusion_matrix(predknn, target_test))
plot_confusion_matrix(knn,data_test,target_test)
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
probNaives = model1.predict_proba(data_test)
fpr, tpr, thresholds = roc_curve(target_test,probNaives[:,1])
roc_auc = auc(fpr, tpr)
ppv = accuracy_score(target_test, pred1, normalize = True)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Naive Bayes')
plt.legend(loc="lower right")
plt.show()

#Multi-layer Perceptron
probMultiLayer = model2.predict_proba(data_test)
fpr, tpr, thresholds = roc_curve(target_test,probMultiLayer[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Multi-layer Perceptron')
plt.legend(loc="lower right")
plt.show()

#k-nn Algorithm
probknn = knn.predict_proba(data_test)
fpr, tpr, thresholds = roc_curve(target_test,probknn[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic k-nn algorithm')
plt.legend(loc="lower right")
plt.show()

#CART Algorithm
probCART = tree_model.predict_proba(data_test)
fpr, tpr, thresholds = roc_curve(target_test,probCART[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic CART algorithm')
plt.legend(loc="lower right")
plt.show()

#Random forest
probRandom = reg.predict_proba(data_test)
fpr, tpr, thresholds = roc_curve(target_test,probRandom[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Random forest')
plt.legend(loc="lower right")
plt.show()

#----------------------------------------------------------------------------------------------------------------------
#G-mean
print("G-mean Naive Bayes:", geometric_mean_score(target_test, predNaive) )
print("G-mean Multi-layer Perceptron:", geometric_mean_score(target_test, predMulti))
print("G-mean k-nn algorithm:", geometric_mean_score(target_test, predknn))
print("G-mean CART algorithm:", geometric_mean_score(target_test, predCart))
print("G-mean Random Forest:", geometric_mean_score(target_test, predRandom))

#----------------------------------------------------------------------------------------------------------------------
#F1-score
print("\nF1-score Naive Bayes:", f1_score(target_test, predNaive) )
print("F1-score Multi-layer Perceptron:", f1_score(target_test, predMulti))
print("F1-score k-nn algorithm:", f1_score(target_test, predknn))
print("F1-score CART algorithm:", f1_score(target_test, predCart))
print("F1-score Random Forest:", f1_score(target_test, predRandom))

#----------------------------------------------------------------------------------------------------------------------
#G-measure

probNaives = model1.predict_proba(data_test)
fpr, tpr, thresholds = roc_curve(target_test,probNaives[:,1])
ppv = precision_score(target_test, predNaive)
tpr = np.array(tpr)
tpr = tpr.tolist()
tpr = list(filter(lambda num: num != 0, tpr))
tpr.append(ppv)
print("\nG-measure Naive Bayes:", gmean(tpr))

probMultiLayer = model2.predict_proba(data_test)
fpr, tpr, thresholds = roc_curve(target_test,probMultiLayer[:,1])
ppv = precision_score(target_test, predMulti)
tpr = np.array(tpr)
tpr = tpr.tolist()
tpr.append(ppv)
tpr.remove(0)
print("G-measure Multi-layer Perceptron:", gmean(tpr))

probknn = knn.predict_proba(data_test)
fpr, tpr, thresholds = roc_curve(target_test,probknn[:,1])
ppv = precision_score(target_test, predknn)
tpr = np.array(tpr)
tpr = tpr.tolist()
tpr.append(ppv)
tpr.remove(0)
print("G-measure k-nn algorithm:", gmean(tpr))

probCART = tree_model.predict_proba(data_test)
fpr, tpr, thresholds = roc_curve(target_test,probCART[:,1])
ppv = precision_score(target_test, predCart)
tpr = np.array(tpr)
tpr = tpr.tolist()
tpr.append(ppv)
tpr.remove(0)
print("G-measure CART algorithm:", gmean(tpr))

probRandom = reg.predict_proba(data_test)
fpr, tpr, thresholds = roc_curve(target_test,probRandom[:,1])
ppv = precision_score(target_test, predRandom)
tpr = np.array(tpr)
tpr = tpr.tolist()
tpr.append(ppv)
tpr.remove(0)
print("G-measure Random Forest:", gmean(tpr))

