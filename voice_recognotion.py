# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 09:18:11 2020
@author: Zeno

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler #To normalize our data 
from sklearn.model_selection import train_test_split # to split our data as train-test
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay # to show confusion matrix (from stackoverflow)

voice_data = pd.read_csv("D:/Machine Learning Works/Logistic Regression/for_Github/voice.csv") # Reading data
# Initially we need to analyze our data to drop NaN features or String features if it has
voice_data.info()
# 20.feature type is Object which contains male or female strings
# Making list comprehension to turn them into integer as 0(female) and 1(male)
voice_data.label = [0 if each== "female" else 1 for each in voice_data.label]
voice_data.info() # Now, every value is numerical 
# Let's get x and y values
y = voice_data.iloc[:,20] # I got Sex column
x = voice_data.drop(["label"],axis = 1) # Drop the label column 
# Let's normalize our data not to other high scaled features dominate low scaled features
x_normalized = StandardScaler().fit(x).transform(x) # I normalized x with StandartScaler method
# We need to split our data as train-test
# I want to use %20 of data for testing and %80 for training
x_test,x_train,y_test,y_train = train_test_split(x_normalized, y, test_size =0.2,random_state = 42)
# Let's create Logistic Model
logistic_model = LogisticRegression()
logistic_model.fit(x_train,y_train)
# Now we can predict 
prediction = logistic_model.predict(x_test)
# For the last step, We can look at our score for accuracy between prediction and real value

#I tried to speculate on the confusion matrix which I found some codes are below from github
cm = confusion_matrix(y_test, prediction, normalize='all')
cmd = ConfusionMatrixDisplay(cm)
cmd.plot()#End of the confusion matrix

print("logistic regression constant : ",logistic_model.intercept_,"\n")
print("logistic regression coefficients : ",logistic_model.coef_,"\n")
print("Accuracy of the prediction is : {}".format(logistic_model.score(x_test,y_test)))

