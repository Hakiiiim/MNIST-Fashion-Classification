# save and load workspaces in python
from __future__ import print_function
import pickle

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import cv2
import skimage.measure as sm

import pandas as pd

import scipy

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# loading:
with open('C:/Users/Asus 6eme/Documents/Data/TP3-Final-cnn/dataset_TP3.pkl','rb') as f:train_images, train_labels = pickle.load(f)

print(train_labels[8542])
class_names = ['Tshirt', 'Pantalon', 'Pull', 'Robe', 'Veste',
               'Sandale', 'Chemise', 'Basket', 'Sac', 'Bottine']

#showing an image from the data set
fig=plt.figure()
n=2175
for i in range(n,n+25):
    a=fig.add_subplot(5,5,i-n+1)
    a.set_title(class_names[train_labels[i]],color='red')
    plt.imshow(train_images[i],cmap='Greys')
plt.show()


#thresholding images (binarize them) 
for i in range(60000):
    train_images[i]=1.0 * (train_images[i] > 16)
    #train_images[i]=1.0 * (train_images[i] > (np.max(train_images[i])/3))
    train_images[i] = abs(train_images[i]-1)
    
    
fig=plt.figure()
n=2175
for i in range(n,n+25):
    a=fig.add_subplot(5,5,i-n+1)
    a.set_title(class_names[train_labels[i]],color='red')
    plt.imshow(train_images[i],cmap='Greys')
plt.show()

I = train_images[6];
Ix = scipy.ndimage.sobel(I ,axis = 0);
Iy = scipy.ndimage.sobel(I ,axis = 1);

fig=plt.figure()
plt.imshow(Ix+Iy)
plt.show()



#################################################
#################################################
#First method : Image processing
#################################################
#################################################


#Extracting geometric descriptors using regionprops
imgdata=np.zeros((60000,9))
for i in range(60000):
    props = sm.regionprops_table(train_images[i], properties=['area', 'bbox_area','convex_area','eccentricity','equivalent_diameter','extent','major_axis_length','minor_axis_length','perimeter'])
    imgdata[i][0]=float(props['area'])
    imgdata[i][1]=float(props['bbox_area'])
    imgdata[i][2]=float(props['convex_area'])
    imgdata[i][3]=float(props['eccentricity'])
    imgdata[i][4]=float(props['equivalent_diameter'])
    imgdata[i][5]=float(props['extent'])
    imgdata[i][6]=float(props['major_axis_length'])
    imgdata[i][7]=float(props['minor_axis_length'])
    imgdata[i][8]=float(props['perimeter'])
    
#Selecting the 5 best features among the 9 computed
imgdata_new = SelectKBest(chi2, k=5).fit_transform(imgdata, train_labels)

#we will convert our data to a dataframe format
df = pd.DataFrame(imgdata_new, columns=['feature1', 'feature2','feature3','feature4','feature5'])

###########################
#Statistical learning
###########################

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(df, train_labels, test_size = 0.25)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

import seaborn as sns; sns.set()

import sklearn.tree
import sklearn.neighbors
from sklearn.svm import SVC # "Support Vector Classifier"
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neural_network

# Defining classification models
logistic_m = LogisticRegression() #linear regression
tree_m = sklearn.tree.DecisionTreeClassifier(max_depth=3) #Decision tree
gradient_descent_m=SGDClassifier()# stochastic gradient descent
gradient_boosting_m=GradientBoostingClassifier()# gradient boosting
knn_m = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
svm_m = SVC(kernel='linear')
neural_net_m = sklearn.neural_network.MLPClassifier(
    hidden_layer_sizes=(8,2),  activation='relu', solver='adam', alpha=0.002, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=2000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08) #Shallow neural network

#Training of the models using the method .fit() 
model1 = logistic_m.fit(X_train, y_train)
model2 = tree_m.fit(X_train, y_train)
model3 = gradient_descent_m.fit(X_train, y_train)
model4 = gradient_boosting_m.fit(X_train, y_train)
model5 = knn_m.fit(X_train, y_train)
model6 = svm_m.fit(X_train, y_train)
model7 = neural_net_m.fit(X_train, y_train)


# predictions using the method .predict()
predictions1 = logistic_m.predict(X_test)
predictions2 = tree_m.predict(X_test)
predictions3 = gradient_descent_m.predict(X_test)
predictions4 = gradient_boosting_m.predict(X_test)
predictions5 = knn_m.predict(X_test)
predictions6 = svm_m.predict(X_test)
predictions7 = neural_net_m.predict(X_test)

#Computing the accuracy of the models
from sklearn.metrics import accuracy_score

accuracy1 = accuracy_score(y_test,predictions1)
accuracy2 = accuracy_score(y_test,predictions2)
accuracy3 = accuracy_score(y_test,predictions3)
accuracy4 = accuracy_score(y_test,predictions4)
accuracy5 = accuracy_score(y_test,predictions5)
accuracy6 = accuracy_score(y_test,predictions6)
accuracy7 = accuracy_score(y_test,predictions7)

print(['Logitic regression',accuracy1])
print(['Decision tree',accuracy2])
print(['Gradient descent',accuracy3])
print(['Gradient boosting',accuracy4])
print(['Knn',accuracy5])
print(['SVM',accuracy6])
print(['Neural network',accuracy7])

#The accuracy's plot
fig=plt.figure()
plt.subplot(1,2,1)
plt.bar([1,2,3,4,5,6,7],height=[accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,accuracy7],tick_label=['Logitic \n regression', 'Decision \n tree', 'Gradient \n descent', 'Gradient \n boosting', 'Knn', 'SVM', 'Neural net'])
plt.title('models accuracy')
plt.ylabel('accuracy')
plt.xlabel('model')


# Re-loading:
with open('C:/Users/Asus 6eme/Documents/Data/TP3-Final-cnn/dataset_TP3.pkl','rb') as f:train_images, train_labels = pickle.load(f)

#visualizing a sample of test data labelized
indexes = X_test.index
fig=plt.figure()
for i in range(0,25):
    a=fig.add_subplot(5,5,i+1)
    a.set_title(class_names[predictions4[i]],color='red')
    plt.imshow(train_images[indexes[i]],cmap='Greys')
plt.show()




###########################################################
###########################################################
#Second method : CNN
###########################################################
###########################################################


from __future__ import print_function
import pickle

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Loading:
with open('C:/Users/Asus 6eme/Documents/Data/TP3-Final-cnn/dataset_TP3.pkl','rb') as f:train_images, train_labels = pickle.load(f)

import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

import os

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size = 0.15)

# Then into validation set
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.1)

batch_size = 128 # You can try 64 or 32 if you'd like to
num_classes = 10
epochs = 10 # loss function value will be stabilized after 93rd epoch# To save the model:
lr = 0.001

input_shape = (28, 28, 1) # 3 dimensional image
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

# Making sure everything is good
print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)
print('X_validation shape: ', X_validation.shape)

#Sequential to concatenate the layers we are going to add
model = Sequential()

#First convolutive layer with 32 Filters 3*3
model.add(Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(0.01), input_shape=input_shape))
model.add(Activation("relu"))

#L'ajout de la Batch normalization permet de limiter le changement dans la distribution des valeurs d'entrée 
# dans un algorithme d'apprentissage (Covariate shift).
# Ceci se fait en normalisant les activations de chaque couche (transformant les entrées en moyenne = 0 et variance = 1). 
#Ceci, permet à chaque couche d’apprendre sur une distribution d’entrées plus stable, ce qui accélérerait la formation du réseau.
model.add(BatchNormalization(axis=-1))

#Dimensionality reduction by replacing with the maximum on each 2*2 window
model.add(MaxPooling2D(pool_size=(2, 2)))
#Dropout 1/4 of the nodes we have in order to avoid overfitting
model.add(Dropout(0.25))

model.add(Flatten())

#A normal dense layer with 64 nodes
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.3))

#And finally the output layer
num_classes = 10
model.add(Dense(num_classes))
model.add(Activation("softmax"))
opt = Adam(lr=lr)

model.compile(loss='categorical_crossentropy', #Now it is time to compile the model specifying the metrics we want to evaluate
              optimizer=opt,
              metrics=['accuracy'])

y_train_c = to_categorical(y_train, num_classes=10, dtype='float32') #Categorizing the labels
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3) # To stop the algorithm if it takes too long

history=model.fit(X_train.astype("float32"), y_train_c,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test.astype("float32"), to_categorical(y_test)),
              shuffle=True,
              callbacks=[early_stop])# It would have been better if we launch the training on a GPU instead

print(history.history)
scores = model.evaluate(X_validation.astype("float32"), to_categorical(y_validation), verbose=0)
print('Validation loss:', scores[0])
print('Validation accuracy:', scores[1])


print(history.history.keys())


fig=plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
fig=plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predictions = model.predict(X_validation.astype("float32"))

class_names = ['Tshirt', 'Pantalon', 'Pull', 'Robe', 'Veste',
               'Sandale', 'Chemise', 'Basket', 'Sac', 'Bottine']
               
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2])

fig=plt.figure()
for i in range(0,25):
    a=fig.add_subplot(5,5,i+1)
    t=predictions[i,:]
    
    if np.argmax(t) == y_validation[i]:
        col='blue'
    else:
        col= 'red'
    
    a.set_title(class_names[np.argmax(t)],color=col)
    plt.imshow(X_validation[i],cmap='Greys')
plt.show()

#Sophisticated visualization of the pourcentage of belonging to the categories assigned by the model to each image
fig=plt.figure()
k=0
for i in range(10,16):
    plt.subplot(3, 4, 2*k+1)
    plt.imshow(X_validation[i],cmap='Greys')
    
    t=predictions[i,:]
    if np.argmax(t) == y_validation[i]:
        col='blue'
    else:
        col= 'red'
    
    plt.xlabel(class_names[np.argmax(t)]+' '+str(int(100*np.max(t)))+'% ('+class_names[y_validation[i]]+')',color=col)  
        
    plt.subplot(3, 4, 2*k+2)
    bar = plt.bar(range(10), t, color="#777777")
    
    bar[np.argmax(t)].set_color('red')
    bar[y_validation[i]].set_color('blue')
    k=k+1
plt.tight_layout()
plt.show()




