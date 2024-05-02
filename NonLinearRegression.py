# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 12:18:09 2019

@author: Suat
"""

import numpy as np
import matplotlib.pyplot as plt

'''
Though Linear regression is very good to solve many problems, 
it cannot be used for all datasets. First recall how linear regression, 
could model a dataset. It models a linear relation between a dependent 
variable y and independent variable x. It had a simple equation, of degree 1, 
for example y =  2x + 3
'''

#x = np.arange(-5.0, 5.0, 0.1)
####numpy.arange([start, ]stop, [step, ]dtype=None)
####Return evenly spaced values within a given interval, numpy array.
#
#
###You can adjust the slope and intercept to verify the changes in the graph
#y = 2*(x) + 3
#y_noise = 2 * np.random.normal(size=x.size)
###Draw random samples from a normal (Gaussian) distribution.
#ydata = y + y_noise
##plt.figure(figsize=(8,6))
#plt.plot(x, ydata,  'bo')
#plt.plot(x,y, 'r') 
#plt.ylabel('Dependent Variable')
#plt.xlabel('Indepdendent Variable')
#plt.show()


'''
Non-linear regressions are a relationship between independent variables x  
and a dependent variable y which result in a non-linear function modeled 
data. Essentially any relationship that is not linear can be termed as 
non-linear, and is usually represented by the polynomial of k degrees 
(maximum power of x).

y = ax^3 + bx^2 + cx + d 

Non-linear functions can have elements like exponentials, logarithms, 
fractions, and others. For example:
y = log(x)
 
Or even, more complicated such as :
y = log(y = ax^3 + bx^2 + cx + d)
 
Let's take a look at a cubic function's graph.'''

#x = np.arange(-5.0, 5.0, 0.1)
#
###You can adjust the slope and intercept to verify the changes in the graph
#y = 1*(x**3) + 1*(x**2) + 1*x + 3
#y_noise = 20 * np.random.normal(size=x.size)
#ydata = y + y_noise
#plt.plot(x, ydata,  'bo')
#plt.plot(x,y, 'r') 
#plt.ylabel('Dependent Variable')
#plt.xlabel('Indepdendent Variable')
#plt.show()


''' Quadratic Function Y = X^2 '''
#x = np.arange(-5.0, 5.0, 0.1)
#
###You can adjust the slope and intercept to verify the changes in the graph
#
#y = np.power(x,2)
#y_noise = 2 * np.random.normal(size=x.size)
#ydata = y + y_noise
#plt.plot(x, ydata,  'bo')
#plt.plot(x,y, 'r') 
#plt.ylabel('Dependent Variable')
#plt.xlabel('Indepdendent Variable')
#plt.show()


'''
Exponential
An exponential function with base c is defined by
Y = a + bc^x
 
where b ≠0, c > 0 , c ≠1, and x is any real number. The base, c, is constant 
and the exponent, x, is a variable.
'''
#X = np.arange(-5.0, 5.0, 0.1)
#
###You can adjust the slope and intercept to verify the changes in the graph
#
#Y= np.exp(X)
#
#plt.plot(X,Y) 
#plt.ylabel('Dependent Variable')
#plt.xlabel('Indepdendent Variable')
#plt.show()


'''Logarithmic'''

#X = np.arange(-5.0, 5.0, 0.1)
#
#Y = np.log(X)
#
#plt.plot(X,Y) 
#plt.ylabel('Dependent Variable')
#plt.xlabel('Indepdendent Variable')
#plt.show()


'''Sigmoidal/Logistic'''

#X = np.arange(-5.0, 5.0, 0.1)
#
#
#Y = 1-4/(1+np.power(3, X-2))
#
#plt.plot(X,Y) 
#plt.ylabel('Dependent Variable')
#plt.xlabel('Indepdendent Variable')
#plt.show()











