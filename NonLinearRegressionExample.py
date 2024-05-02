# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 12:49:05 2019

@author: Suat
"""

'''
We're going to try and fit a non-linear model to the datapoints 
corresponding to China's GDP from 1960 to 2014. We download a dataset with two 
columns, the first, a year between 1960 and 2014, the second, China's 
corresponding annual gross domestic income in US dollars for that year.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv")
df.head(10)

'''Plotting the Dataset
This is what the datapoints look like. It kind of looks like an either 
logistic or exponential function. The growth starts off slow, then from 2005 
on forward, the growth is very significant. And finally, it decelerate 
slightly in the 2010s.'''

plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


'''Choosing a model
From an initial look at the plot, we determine that the logistic function 
could be a good approximation, since it has the property of starting with a 
slow growth, increasing growth in the middle, and then decreasing again at 
the end; as illustrated below:
'''

#X = np.arange(-5.0, 5.0, 0.1)
#Y = 1.0 / (1.0 + np.exp(-X))
#
#plt.plot(X,Y) 
#plt.ylabel('Dependent Variable')
#plt.xlabel('Indepdendent Variable')
#plt.show()



'''Building The Model'''

def sigmoid(x, Beta_1, Beta_2): #logistic = sigmoid
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

#Lets look at a sample sigmoid line that might fit with the data:
beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')

##Our task here is to find the best parameters for our model. Lets first 
##normalize our x and y:
xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

'''How we find the best parameters for our fit line?'''
'''
We can use curve_fit which uses non-linear least squares to fit our 
sigmoid function, to data. Optimal values for the parameters so that the 
sum of the squared residuals of sigmoid(xdata, *popt) - ydata is minimized.

popt are our optimized parameters.
'''

from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))


x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


'''Calculate the accuracy of the model'''
# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )


