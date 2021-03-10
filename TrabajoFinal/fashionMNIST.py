#https://www.tensorflow.org/tutorials/keras/classification
#TensorFlow y tf.keras
from scipy.sparse import data
from sklearn.utils import validation
import tensorflow as tf
from tensorflow import keras

#Librerias de ayuda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Modelo de ensembles
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Modelo de Deep learning
from tensorflow.python.keras.layers import Dense, Flatten

#Cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#Procesado de imágenes
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator

#https://github.com/zalandoresearch/fashion-mnist
#https://www.tensorflow.org/tutorials/keras/classification
#loading data with tensorflow

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Mostramos una y varias figuras del conjunto de entrenamiento
#para ver si los datos se han pasado adecuadamente
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.savefig('ropa1.jpg')


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.savefig('conjuntoropa.jpg')

#Mostrar cuantos datos hay de cada conjunto
plt.figure()
unique, counts = np.unique(train_labels, return_counts=True)
sns.barplot(x = unique, y = counts)
plt.ylabel('Digit Frequency')
plt.xlabel('Digit')
plt.savefig('labels.jpg')

#----------------------------------------------------------------------------------------------------------------------------
#Procesado de imágenes para Machine Learning

train_images1 = train_images.flatten().reshape(60000, 784)
test_images1 = test_images.flatten().reshape(10000,784)


#Procesado de imágenes para Deep Learning
def change_size(image):
    img = array_to_img(image, scale=False) #returns PIL Image
    img = img.resize((75, 75)) #resize image
    img = img.convert(mode='RGB') #makes 3 channels
    arr = img_to_array(img) #convert back to array
    return arr.astype(np.float64)

train_arr = np.array(train_images).reshape(-1, 28, 28, 1)
train_i = [change_size(img) for img in train_arr]
del train_arr
train_i = np.array(train_i)

train_l = tf.keras.utils.to_categorical(train_labels)

#----------------------------------------------------------------------------------------------------------------------------
#Modelo de DEEP LEARNING PRE-ENTRENADO
#https://towardsdatascience.com/transfer-learning-using-pre-trained-alexnet-model-and-fashion-mnist-43898c2966fb
#https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751#:~:text=A%20pre%2Dtrained%20model%20is,VGG%2C%20Inception%2C%20MobileNet).
#https://keras.io/api/applications/
#https://wngaw.github.io/transfer-learning-for-image-classification/
#http://personal.cimat.mx:8181/~mrivera/cursos/aprendizaje_profundo/preentrenadas/preentrenadas.html
#http://www.eamonfleming.com/projects/fashion-mnist.html
#https://www.kaggle.com/saumandas/intro-to-transfer-learning-with-mnist

image_gen = ImageDataGenerator(rescale=1./255,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              validation_split=0.2) 


train_generator = image_gen.flow(train_i, 
                                 train_l,
                                batch_size=32,
                                shuffle=True,
                                subset='training',
                                seed=42)

valid_generator = image_gen.flow(train_i,
                                 train_l,
                                batch_size=16,
                                shuffle=True,
                                subset='validation')
del train_i #saves RAM

model = keras.Sequential()

# Resnet50 loss: 0.3512 - accuracy: 0.8712 - val_loss: 0.3220 - val_accuracy: 0.8831
# Epoch 5
# InceptionV3 loss: 0.2446 - accuracy: 0.9125 - val_loss: 0.2267 - val_accuracy: 0.9180
# Epoch 5
model.add(tf.keras.applications.inception_v3.InceptionV3(input_shape = (75, 75, 3), 
                                include_top = False, 
                                weights = 'imagenet'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
for layer in model.layers[0].layers:
    if layer.name == 'conv5_block1_0_conv':
        break
    layer.trainable=False

history = model.fit(train_generator, validation_data=valid_generator, epochs=5, 
          steps_per_epoch=train_generator.n//train_generator.batch_size,
         validation_steps=valid_generator.n//valid_generator.batch_size)

plt.figure()  
plt.plot(history.history['accuracy'],'r')  
plt.plot(history.history['val_accuracy'],'g')  
plt.xticks(np.arange(0, 6, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Accuracy")  
plt.title("Training Accuracy vs Validation Accuracy")  
plt.legend(['train','validation'])
plt.savefig('InceptionV3_accuracy.jpg')

plt.figure()  
plt.plot(history.history['loss'],'r')  
plt.plot(history.history['val_loss'],'g')  
plt.xticks(np.arange(0, 6, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss")  
plt.legend(['train','validation'])
plt.savefig('InceptionV3_loss.jpg')  


#----------------------------------------------------------------------------------------------------------------------------
# Clasificadores clásicos de Machine Learning
#https://github.com/graydenshand/FASHION-MNIST-CLASSIFICATION__Random-Forest-vs-Multi-Layer-Perceptron/blob/master/Fashion_MNIST.py
#https://www.kaggle.com/gpreda/cnn-with-tensorflow-keras-for-fashion-mnist

#MODELO DE ENSEMBLES

train_images = train_images / 255.0

test_images = test_images / 255.0

RandomForest = RandomForestClassifier(n_estimators=100)

RandomForest.fit(train_images1, train_labels)

print("Random Forest accuracy:", RandomForest.score(test_images1, test_labels))

#Cross validation Random Forest
cv_scores = cross_val_score(RandomForest, train_images1, train_labels, cv=5)

#print each cv score (accuracy) and average them
print(cv_scores)
print('Random Forest cv_scores mean:{}'.format(np.mean(cv_scores)))

del train_images1
del test_images1
#----------------------------------------------------------------------------------------------------------------------------
#  https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
# Aproximaciones basadas en Deep Learning
#MODELO SECUENCIAL

#Unimos entradas
inputs = np.concatenate((train_images, test_images), axis=0)
targets = np.concatenate((train_labels, test_labels), axis=0)

acc = []
loss = []

# KFold para la validación cruzada
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

  Secuencial = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(10, activation='softmax'),
  ])

  Secuencial.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  Secuencial_model = Secuencial.fit(inputs[train], targets[train],
                    batch_size=32,
                    epochs=20,
                    verbose=1, validation_data = (inputs[test], targets[test]))


  print(Secuencial.summary())
  test_loss, test_acc = Secuencial.evaluate(inputs[test],  targets[test], verbose=2)

  #Precisión del modelo
  acc.append(test_acc)
  loss.append(test_loss)

  plt.figure()  
  plt.plot(Secuencial_model.history['accuracy'],'r')  
  plt.plot(Secuencial_model.history['val_accuracy'],'g')
  plt.xticks(np.arange(0, 21, 2.0))  
  plt.rcParams['figure.figsize'] = (8, 6)  
  plt.xlabel("Num of Epochs")  
  plt.ylabel("Accuracy")  
  plt.title("Training Accuracy vs Validation Accuracy")  
  plt.legend(['train','validation'])
  plt.savefig('Secuencial_accuracy{}.jpg'.format(fold_no))  

  plt.figure()  
  plt.plot(Secuencial_model.history['loss'],'r')  
  plt.plot(Secuencial_model.history['val_loss'],'g')  
  plt.xticks(np.arange(0, 21, 2.0))  
  plt.rcParams['figure.figsize'] = (8, 6)  
  plt.xlabel("Num of Epochs")  
  plt.ylabel("Loss")  
  plt.title("Training Loss vs Validation Loss")  
  plt.legend(['train','validation'])
  plt.savefig('Secuencial_loss{}.jpg'.format(fold_no))  

  fold_no = fold_no + 1

print(acc)
print('Secuencial Model cv accuracy mean:{}'.format(np.mean(acc)))

print(loss)
print('Secuencial Model cv loss mean:{}'.format(np.mean(loss)))

