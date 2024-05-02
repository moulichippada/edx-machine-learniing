# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 20:51:30 2019

@author: Suat
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


df = pd.read_csv(r"https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv")

## take a look at the dataset
df.head()


##Lets select some features that we want to use for regression.
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY',
          'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


##Lets plot Emission values with respect to Engine size:
#plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
#plt.xlabel("Engine size")
#plt.ylabel("Emission")
#plt.show()


'''Creating train and test dataset
    note: Please see SİmpleLinearRegression file for train and data set
    explanations'''
    
msk = np.random.rand(len(df)) < 0.8 #80% of data selected as train set
train = cdf[msk]
test = cdf[~msk]

'''Train data distribution'''
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


'''Multiple Regression Model'''
'''In reality, there are multiple variables that predict the Co2emission. 
When more than one independent variable is present, the process is called 
multiple linear regression. For example, predicting co2emission using 
FUELCONSUMPTION_COMB, EngineSize and Cylinders of cars. 
The good thing here is that Multiple linear regression is 
the extension of simple linear regression model.'''

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

''' Coefficient and Intercept , are the parameters of the fit line. 
Given that it is a multiple linear regression, with 3 parameters, and 
knowing that the parameters are the intercept and coefficients of hyperplane, 
sklearn can estimate them from our data. Scikit-learn uses plain 
Ordinary Least Squares method to solve this problem.'''

'''Ordinary Least Squares (OLS)'''
'''OLS is a method for estimating the unknown parameters in a linear 
regression model. OLS chooses the parameters of a linear function of a set of 
explanatory variables by minimizing the sum of the squares of the differences 
between the target dependent variable and those predicted by the linear
function. In other words, it tries to minimizes the sum of squared 
errors (SSE) or mean squared error (MSE) between the target variable (y) 
and our predicted output (y^) over all samples in the dataset.

OLS can find the best parameters using of the following methods:
    - Solving the model parameters analytically using closed-form equations
    - Using an optimization algorithm (Gradient Descent, Stochastic Gradient 
    Descent, Newton’s Method, etc.)'''

'''PREDICTION'''
y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares - 3VAR: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score for 3VAR: %.2f' % regr.score(x, y))


##multiple linear regression with another variables
##Instead of using combination, using city and highway fuelconsumption
##values. Will it change the results? Hint: NO!

regr2 = linear_model.LinearRegression()
z = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY',
                           'FUELCONSUMPTION_HWY']])
q = y = np.asanyarray(train[['CO2EMISSIONS']])
regr2.fit (z, q)


q_hat = regr2.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY',
                           'FUELCONSUMPTION_HWY']])
z = np.asanyarray(test[['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_CITY',
                           'FUELCONSUMPTION_HWY']])
q = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares - 4VAR: %.2f"
      % np.mean((q_hat - q) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score for 4VAR: %.2f' % regr2.score(z, q))



    