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
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
#import train data
train_datagen = ImageDataGenerator(rescale=1/255,
                                   shear_range = 0.3,
                                   zoom_range = 0.3,horizontal_flip = True,
                                   vertical_flip =  True , 
                                   rotation_range=60,
                                   brightness_range = (0.4, 1.4))


train_data = train_datagen.flow_from_directory('../input/soil-classification-image-data/Soil_Dataset/Train',
                                                 target_size = (244, 244),
                                                 class_mode='categorical',
                                                 shuffle=True,seed=1)
#import test data

test_datagen = ImageDataGenerator(rescale = 1/255)
test_data = test_datagen.flow_from_directory("../input/soil-classification-image-data/Soil_Dataset/Test",
                                                           target_size=(244,244),
                                                           class_mode='categorical',
                                                           shuffle=True,seed=1)
test_labels = test_data.classes 
 
# convert the training labels to categorical vectors 
test_labels = to_categorical(test_labels, num_classes=4)
# Instanciate an empty model
model = Sequential()

# Adding a Convolution Layer C1
# Input shape = N = (28 x 28)
# No. of filters  = 6
# Filter size = f = (5 x 5)
# Padding = P = 0
# Strides = S = 1
# Size of each feature map in C1 is (N-f+2P)/S +1 = 28-5+1 = 24
# No. of parameters between input layer and C1 = (5*5 + 1)*6 = 156
model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', input_shape=(244,244,3), activation='tanh'))

# Adding an Average Pooling Layer S2
# Input shape = N = (24 x 24)
# No. of filters = 6
# Filter size = f = (2 x 2)
# Padding = P = 0
# Strides = S = 2
# Size of each feature map in S2 is (N-f+2P)/S +1 = (24-2+0)/2+1 = 11+1 = 12
# No. of parameters between C1 and S2 = (1+1)*6 = 12
model.add(AvgPool2D(pool_size=(2,2)))

# Adding a Convolution Layer C3
# Input shape = N = (12 x 12)
# No. of filters  = 16
# Filter size = f = (5 x 5)
# Padding = P = 0
# Strides = S = 1
# Size of each feature map in C3 is (N-f+2P)/S +1 = 12-5+1 = 8
# No. of parameters between S2 and C3 = (5*5*6*16 + 16) + 16 = 2416
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))

# Adding an Average Pooling Layer S4
# Input shape = N = (8 x 8)
# No. of filters = 16
# Filter size = f = (2 x 2)
# Padding = P = 0
# Strides = S = 2
# Size of each feature map in S4 is (N-f+2P)/S +1 = (8-2+0)/2+1 = 3+1 = 4
# No. of parameters between C3 and S4 = (1+1)*16 = 32
model.add(AvgPool2D(pool_size=(2,2)))

# As compared to LeNet-5 architecture there was one more application of convolution but in our code  further application of 
# convolution with (5 x 5) filter would result in a negative dimension which is not possible. So we aren't applying any more
# convolution here.

# Flattening the layer S4
# There would be 16*(4*4) = 256 neurons
model.add(Flatten())

# Adding a Dense layer with `tanh` activation+# 
# No. of inputs = 256
# No. of outputs = 120
# No. of parameters = 256*120 + 120 = 30,840
model.add(Dense(120, activation='tanh'))

# Adding a Dense layer with `tanh` activation
# No. of inputs = 120
# No. of outputs = 84
# No. of parameters = 120*84 + 84 = 10,164
model.add(Dense(84, activation='tanh'))

# Adding a Dense layer with `softmax` activation
# No. of inputs = 84
# No. of outputs = 10
# No. of parameters = 84*10 + 10 = 850
model.add(Dense(4, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.fit(train_data, epochs=20, verbose=1, batch_size=32,validation_data=test_data)
score = model.evaluate(test_data)

print('Test Loss:', score[0])
print('Test accuracy:', score[1])
train_datagen1 = ImageDataGenerator(rescale=1/255,
                                   shear_range = 0.3,
                                   zoom_range = 0.3,horizontal_flip = True,
                                   vertical_flip =  True , 
                                   rotation_range=60,
                                   brightness_range = (0.4, 1.4))


train_data1 = train_datagen1.flow_from_directory('../input/soil-classification-image-data/Soil_Dataset/Train',
                                                 target_size = (244, 244),
                                                 class_mode='sparse',
                                                 shuffle=True,seed=1)

test_datagen1 = ImageDataGenerator(rescale = 1/255)
test_data1 = test_datagen1.flow_from_directory("../input/soil-classification-image-data/Soil_Dataset/Test",
                                                           target_size=(244,244),
                                                           class_mode='sparse',
                                                           shuffle=True,seed=1)
X_train, y_train = next(train_data1)
X_test, y_test = next(test_data1)
preds = model.predict(X_test)
preds = np.argmax(preds,axis=1)
from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm=metrics.classification_report(y_test,preds)
print(cm)