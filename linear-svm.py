import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import os
from tqdm import tqdm
DATADIR = '../input/soil-classification-image-data/Soil_Dataset/Train'
DATADIR1 = '../input/soil-classification-image-data/Soil_Dataset/Test'
CATEGORIES = ['Alluvial_Soil','Black_Soil','Clay_Soil','Red_Soil']
IMG_SIZE=100
for category in CATEGORIES:
    path=os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        plt.imshow(img_array)
        plt.show()
        break
    break
training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()        
testing_data=[]
def create_testing_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR1, category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                testing_data.append([new_array,class_num])
            except Exception as e:
                pass
create_testing_data()      
print(len(training_data))
print(len(testing_data))
lenofimage = len(training_data)
lenoftest = len(testing_data)
X=[]
y=[]
X_test=[]
y_test=[]
for categories, label in training_data:
    X.append(categories)
    y.append(label)
X= np.array(X).reshape(lenofimage,-1)
for categories, label in testing_data:
    X_test.append(categories)
    y_test.append(label)
X_test= np.array(X_test).reshape(lenoftest,-1)
X = X/255.0
X_test = X_test/255.0
y=np.array(y)
y_test=np.array(y_test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =X,X_test,y,y_test
from sklearn.svm import SVC
svc = SVC(kernel='linear',gamma='auto')
svc.fit(X_train, y_train)
y2 = svc.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy on unknown data is",accuracy_score(y_test,y2))
result = pd.DataFrame({'original' : y_test,'predicted' : y2})