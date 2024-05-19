import os
import cv2
import numpy as np
import pandas as pd
import random
import pickle

# creating list of labels
dataset=[]
path = "dataset_full"
label_list = os.listdir(path)

#Importing training data
for i in os.listdir(path):
    in_path = os.path.join(path,i)
    c = 0
    for j in os.listdir(in_path):
        img = cv2.imread(os.path.join(in_path,j))
        dataset.append([img,label_list.index(i),str(j)])
        c+=1
        if c>=500:
            break

#shuffling for better performance
random.Random(56).shuffle(dataset)

# Extracting image by image
Texture=[]
Shape=[]
Color=[]
label=[]

from feature_extract import Texture_Extraction, Shape_Extraction, Color_Extraction
for i in dataset:
    texture_feature = Texture_Extraction(i[0])
    shape_feature = Shape_Extraction(i[0])
    color_feature = Color_Extraction(i[0])
    label.append(i[1])
    Texture.append(texture_feature)
    Shape.append(shape_feature)
    Color.append(color_feature)

#Merging all together
Texture=np.array(Texture)
Shape=np.array(Shape)
Color=np.array(Color)
final_data = np.concatenate((Texture,Shape,Color),axis=1)

#Machine Learning Implementation

#Creating Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(final_data,label,test_size=0.2,random_state=150)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
Y_train = pd.Series(Y_train)
Y_test = pd.Series(Y_test)

X_train.shape , X_test.shape , Y_train.shape , Y_test.shape

#Traing with best model

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=110, min_samples_split=40, min_impurity_decrease=0.0)
model.fit(X_train,Y_train)

print(model.score(X_train,Y_train))
print(model.score(X_test,Y_test))
print(model.score(final_data,label))

#Saving the model
#pickle.dump(model, open("static/model/model.pkl", 'wb'))