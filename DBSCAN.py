# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 00:11:37 2019

@author: Suat
Density Based Clustering
"""
"""
Density-Based Clustering
Most of the traditional clustering techniques, such as k-means, hierarchical 
and fuzzy clustering, can be used to group data without supervision.

However, when applied to tasks with arbitrary shape clusters, or clusters 
within cluster, the traditional techniques might be unable to achieve good 
results. That is, elements in the same cluster might not share enough 
similarity or the performance may be poor. Additionally, Density-based 
Clustering locates regions of high density that are separated from one 
another by regions of low density. Density, in this context, is defined
as the number of points within a specified radius.

In this code, the main focus will be manipulating the data and 
properties of DBSCAN and observing the resulting clustering.
"""

import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 

"""
Data generation
The function below will generate the data points and requires these inputs:
    
* centroidLocation: Coordinates of the centroids that will generate the 
  random data.
      Example: input: [[4,3], [2,-1], [-1,4]]
      
* numSamples: The number of data points we want generated, split over 
  the number of centroids (# of centroids defined in centroidLocation)
      Example: 1500
      
* clusterDeviation: The standard deviation between the clusters. 
  The larger the number, the further the spacing.
      Example: 0.5
"""

def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    # Create random data and store in feature matrix X and response vector y.
    X, y = make_blobs(n_samples=numSamples, centers=centroidLocation, 
                                cluster_std=clusterDeviation)
    
    # Standardize features by removing the mean and scaling to unit variance
    X = StandardScaler().fit_transform(X)
    return X, y

##Use createDataPoints with the 3 inputs and store the output into 
##variables X and y.
X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)

"""
Modeling
DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. 
This technique is one of the most common clustering algorithms which works 
based on density of object. The whole idea is that if a particular point 
belongs to a cluster, it should be near to lots of other points in that 
cluster.

It works based on two parameters: Epsilon and Minimum Points

Epsilon determine a specified radius that if includes enough number of 
points within, we call it dense area

minimumSamples determine the minimum number of data points we want in a 
neighborhood to define a cluster.
"""

epsilon = 0.3
minimumSamples = 7
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_
labels

"""
Distinguishing Outliers
Lets Replace all elements with 'True' in core_samples_mask that are in the 
cluster, 'False' if the points are outliers.
"""

# First, create an array of booleans using the labels from db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_

# Remove repetition in labels by turning it into a set.
unique_labels = set(labels)
unique_labels


"""
Data visualization
"""

# Create colors for the clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
colors

# Plot the points with colors
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    # Plot the datapoints that are clustered
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=col, marker=u'o', alpha=0.5)

    # Plot the outliers
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=col, marker=u'o', alpha=0.5)


"""
Practice
To better underestand differences between partitional and density-based 
clusteitng, try to cluster the above dataset into 3 clusters 
using k-Means.
Notice: do not generate data again, use the same dataset as above.
"""

# write your code here

from sklearn.cluster import KMeans 
k = 3
k_means3 = KMeans(init = "k-means++", n_clusters = k, n_init = 12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(k), colors):
    my_members = (k_means3.labels_ == k)
    plt.scatter(X[my_members, 0], X[my_members, 1],  c=col, marker=u'o', alpha=0.5)
plt.show()




    




