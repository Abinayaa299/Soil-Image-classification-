import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import kerastuner as kt
from tensorflow import keras
import tensorflow as tf
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameter as hp
from keras.layers import Dense,Dropout,Activation,Add,MaxPooling2D,Conv2D,Flatten,AvgPool2D
from keras.models import Sequential 
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG19
from keras import layers
from keras.preprocessing import image
import matplotlib.pyplot as plt
import seaborn as sns
train_datagen = ImageDataGenerator(rescale=1/255,
                                   shear_range = 0.3,
                                   zoom_range = 0.3,horizontal_flip = True,
                                   vertical_flip =  True , 
                                   rotation_range=60,
                                   brightness_range = (0.4, 1.4))


train_data = train_datagen.flow_from_directory('../input/soil-classification-image-data/Soil_Dataset/Train',
                                                 target_size = (244, 244),
                                                 class_mode='sparse',
                                                 shuffle=True,seed=1)

test_datagen = ImageDataGenerator(rescale = 1/255)
test_data = test_datagen.flow_from_directory("../input/soil-classification-image-data/Soil_Dataset/Test",
                                                           target_size=(244,244),
                                                           class_mode='sparse',
                                                           shuffle=True,seed=1)
## Defining Cnn
model = tf.keras.models.Sequential([
  layers.Conv2D(32, 3, activation='relu',input_shape=(244,244,3)),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.3),
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dropout(0.15),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.1),
  layers.Dense(4, activation= 'softmax')
])
model.summary()
#to avoid overfitting
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
## fit model
FH=model.fit(train_data,validation_data= test_data,batch_size=32,epochs = 100,callbacks=[early])
#evaluate model
model.evaluate(test_data)
y_pred =  model.predict(test_data)
y_pred =  np.argmax(y_pred,axis=1)
y_pred
#plotting training values
sns.set()

acc = FH.history['accuracy']
val_acc = FH.history['val_accuracy']
loss = FH.history['loss']
val_loss = FH.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
class_names = {2:"Clay_Soil",1:"Black_Soil",3:"RED_Soil",0:"ALLUVIAL_Soil"}
#visulaize data
fig, ax = plt.subplots()
ax.bar("alluvial",175+48,color="r",label="alluvial")
ax.bar("black",47+212,color="b",label="black")
ax.bar("clay",144+47,color="g",label="clay")
ax.bar("red",184+46,color="y",label="red")
ax.legend()
test_labels = test_data.classes 
from keras.utils import to_categorical
# convert the training labels to categorical vectors 
test_labels = to_categorical(test_labels, num_classes=4)
X_train, y_train = next(train_data)
X_test, y_test = next(test_data)
preds = model.predict(X_test)
preds = np.argmax(preds,axis=1)
from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm=metrics.classification_report(y_test,preds)
print(cm)
y_test
image_path = "../input/soil-classification-image-data/Soil_Dataset/Test/Red_Soil/Red_35.jpg"
new_img = image.load_img(image_path, target_size=(244, 244))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
prediction = np.argmax(prediction,axis=1)
print(prediction)
print(class_names[prediction[0]])
plt.imshow(new_img)
