# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:18:58 2019

@author: Suat
"""

"""
Weather Station Clustering using DBSCAN & scikit-learn 
DBSCAN is specially very good for tasks like class identification on a 
spatial context. The wonderful attribute of DBSCAN algorithm is that it can 
find out any arbitrary shape cluster without getting affected by noise. 
For example, this following example cluster the location of weather stations 
in Canada. 

DBSCAN can be used here, for instance, to find the group of stations which 
show the same weather condition. As you can see, it not only finds different 
arbitrary shaped clusters, can find the denser part of data-centered samples 
by ignoring less-dense areas or noises.

let's start playing with the data. We will be working according to the 
following workflow:
"""


import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 

"""
About the dataset
Environment Canada Monthly Values for July - 2015 
"""

"""
Name in the table	 -------------  Meaning
Stn_Name	        -------------  Station Name
Lat	Latitude      --------------  (North+, degrees)
Long	           -------------- Longitude (West - , degrees)
Prov	           -------------- Province
Tm	              -------------- Mean Temperature (°C)
DwTm	          --------------  Days without Valid Mean Temperature
D	----------------   Mean Temperature difference from Normal (1981-2010) (°C)
Tx	              --------------  Highest Monthly Maximum Temperature (°C)
DwTx	--------------- Days without Valid Maximum Temperature
Tn	--------------  Lowest Monthly Minimum Temperature (°C)
DwTn	--------------  Days without Valid Minimum Temperature
S	---------------    Snowfall (cm)
DwS	---------------     Days without Valid Snowfall
S%N	-----------------   Percent of Normal (1981-2010) Snowfall
P	-------------------    Total Precipitation (mm)
DwP	--------------------   Days without Valid Precipitation
P%N	------------------     Percent of Normal (1981-2010) Precipitation
S_G	-----------------------  Snow on the ground at the end of the month (cm)
Pd	--------------------    Number of days with Precipitation 1.0 mm or more
BS	----------------------   Bright Sunshine (hours)
DwBS ---------------------   Days without Valid Bright Sunshine
BS%	---------------------   Percent of Normal (1981-2010) Bright Sunshine
HDD	------------------------  Degree Days below 18 °C
CDD	--------------------------  Degree Days above 18 °C
Stn_No	----------------   Climate station identifier (first 3 digits indicate drainage basin, last 4 characters are for sorting alphabetically).
NA	-------------------------    Not Available
"""

import pandas as pd

filename='weather-stations20140101-20141231.csv'

#Read csv
pdf = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/weather-stations20140101-20141231.csv")
pdf.head(5)

"""
3-Cleaning
Lets remove rows that don't have any value in the Tm field.
"""

pdf = pdf[pd.notnull(pdf["Tm"])]
pdf = pdf.reset_index(drop=True)
pdf.head(5)


"""
4-Visualization
Visualization of stations on map using basemap package. The matplotlib 
basemap toolkit is a library for plotting 2D data on maps in Python. 
Basemap does not do any plotting on it’s own, but provides the facilities 
to transform coordinates to a map projections. 

Please notice that the size of each data points represents the average of 
maximum temperature for each station in a year.
"""
from mpl_toolkits.basemap import Basemap
from pylab import rcParams

rcParams['figure.figsize'] = (14,10)

llon=-140
ulon=-50
llat=40
ulat=65

pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) &(pdf['Lat'] < ulat)]

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

# To collect data based on stations        

xs,ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
pdf['xm']= xs.tolist()
pdf['ym'] =ys.tolist()

#Visualization1
for index,row in pdf.iterrows():
#   x,y = my_map(row.Long, row.Lat)
   my_map.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
#plt.text(x,y,stn)
plt.show()

"""
5- Clustering of stations based on their location i.e. Lat & Lon
DBSCAN form sklearn library can runs DBSCAN clustering from vector array or 
distance matrix.
In our case, we pass it the Numpy array Clus_dataSet to find core samples 
of high density and expands clusters from them.
"""
import sklearn.utils
sklearn.utils.check_random_state(1000)
Clus_dataSet = pdf[['xm','ym']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN
db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels)) 


# A sample of clusters
pdf[["Stn_Name","Tx","Tm","Clus_Db"]].head(5)

set(labels)

"""
6- Visualization of clusters based on location
Now, we can visualize the clusters using basemap:
"""
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
#my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

# To create a color map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))



#Visualization1
for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]                    
    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm) 
        ceny=np.mean(clust_set.ym) 
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))

"""
7- Clustering of stations based on their location, mean, max, and min Temperature
In this section we re-run DBSCAN, but this time on a 5-dimensional dataset:
"""
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataSet = pdf[['xm','ym','Tx','Tm','Tn']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels)) 


# A sample of clusters
pdf[["Stn_Name","Tx","Tm","Clus_Db"]].head(5)


"""
8- Visualization of clusters based on location and Temperature
"""
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
#my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

# To create a color map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))



#Visualization1
for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]                    
    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm) 
        ceny=np.mean(clust_set.ym) 
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))



