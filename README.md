# Soil Image Classification

## Table of contents
* [Introduction](#introduction)
* [Dataset description](#dataset-description)
* [Soil Classification with CNN](#soil-classification-with-cnn)
* [Details of other modules](#details-of-other-modules)
* [Comparison of models](#comparison-of-models)
* [Technologies](#technologies)

## Introduction
This project inputs soil images from the user and states the type of the soil as output. SVM and CNN architectures like LeNet, AlexNet, VGG 16, ResNet are used for soil image classification and evaluated the accuracy of each of the classifiers.
The reason for their performance has been analysed and presented.

## Dataset description
The dataset is obtained from the following [kaggle URL](https://www.kaggle.com/omkargurav/soil-classification-image-data).
The data set consists of 903 RGB images labelled as "Alluvial Soil", "Red Soil", "Clay Soil", "Black Soil".


## Soil Classification with CNN
While there are many predefined CNN models available, a custom CNN model has been developed to accommodate for the soil images dataset. 
Due to the low number of images in the dataset, data augmentation is done. Then the CNN model is created and trained using the training dataset. The number of epochs for training is varying since callback early function is used. Hence if there is no more improvement in the  loss parameter, the epochs are terminated. Maximum number of epochs is fixed as 100. 
Images are converted to 244 X 244 size with RGB color values. The colors are retained as they are important in soil classification.  Adam optimisers are used to adjust the weight parameters. Accuracy is monitored with Sparse categorical cross entropy function.

## Details of other modules
![Other models](https://github.com/Abinayaa299/Soil-Image-classification-/blob/main/details.PNG)

## Comparison of models
![Compare models](https://github.com/Abinayaa299/Soil-Image-classification-/blob/main/compare.PNG)

## Technologies
Project is created with:
* Tensorflow
* Python
