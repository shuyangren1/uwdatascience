#!/usr/bin/env python
# coding: utf-8

# My dataset for Milestone 3 is the same one I used for 2 that I got from archive.ics.uci.edu. The data set is a census report regarding to adults that make above 50k a year (and those who don't). It has 32560 observations, 15 attributes including numeric and qualitative (I will show dtypes below). 
# 
# On a side note, I was snowed in for the lecture on this week's lesson and I have a hard time understanding the materials from just what I learned through going over the recording. I was wondering if I can get a look at a proper solution to better understand what I am suppose to be doing. I understand all the concept pretty well but I am very confused about which methods to call and how to set up the arguments.

# In[98]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Loading in methods for the asisgnment

def FindLabelOfClosest(Points, ClusterCentroids): # determine Labels from Points and ClusterCentroids
    NumberOfClusters, NumberOfDimensions = ClusterCentroids.shape # dimensions of the initial Centroids
    Distances = np.array([float('inf')]*NumberOfClusters) # centroid distances
    NumberOfPoints, NumberOfDimensions = Points.shape
    Labels = np.array([-1]*NumberOfPoints)
    for PointNumber in range(NumberOfPoints): # assign labels to all data points            
        for ClusterNumber in range(NumberOfClusters): # for each cluster
            # Get distances for each cluster
            Distances[ClusterNumber] = np.sqrt(sum((Points.loc[PointNumber,:] - ClusterCentroids.loc[ClusterNumber,:])**2))                
        Labels[PointNumber] = np.argmin(Distances) # assign to closest cluster
    return Labels # return the a label for each point

def CalculateClusterCentroid(Points, Labels): # determine centroid of Points with the same label
    ClusterLabels = np.unique(Labels) # names of labels
    NumberOfPoints, NumberOfDimensions = Points.shape
    ClusterCentroids = pd.DataFrame(np.array([[float('nan')]*NumberOfDimensions]*len(ClusterLabels)))
    for ClusterNumber in ClusterLabels: # for each cluster
        # get mean for each label 
        ClusterCentroids.loc[ClusterNumber, :] = np.mean(Points.loc[ClusterNumber == Labels, :])
    return ClusterCentroids # return the a label for each point

def KMeans(Points, ClusterCentroidGuesses):
    ClusterCentroids = ClusterCentroidGuesses.copy()
    Labels_Previous = None
    # Get starting set of labels
    Labels = FindLabelOfClosest(Points, ClusterCentroids)
    while not np.array_equal(Labels, Labels_Previous):
        # Re-calculate cluster centers based on new set of labels
        ClusterCentroids = CalculateClusterCentroid(Points, Labels)
        Labels_Previous = Labels.copy() # Must make a deep copy
        # Determine new labels based on new cluster centers
        Labels = FindLabelOfClosest(Points, ClusterCentroids)
    return Labels, ClusterCentroids

def Plot2DKMeans(Points, Labels, ClusterCentroids, Title):
    for LabelNumber in range(max(Labels)+1):
        LabelFlag = Labels == LabelNumber
        color =  ['c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y'][LabelNumber]
        marker = ['s', 'o', 'v', '^', '<', '>', '8', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'][LabelNumber]
        plt.scatter(Points.loc[LabelFlag,0], Points.loc[LabelFlag,1],
                    s= 100, c=color, edgecolors="black", alpha=0.3, marker=marker)
        plt.scatter(ClusterCentroids.loc[LabelNumber,0], ClusterCentroids.loc[LabelNumber,1], s=200, c="black", marker=marker)
    plt.title(Title)
    plt.show()


# In[102]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" # data set of income/adults
inc = pd.read_csv(url, sep=", ", header=None, engine="python")
# assign reasonable names to columns
inc.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income_above_50K?"]
inc.head()


# In[103]:


inc.dtypes
# as you can see it the dataset has quantative attributes such as age, years of education, and captial gain/losses.
# It also has binary questions such as "income_above_50K?". While stored as object, it can easily be transformed into a boolean
# From my observation and analysis on the dataset in my last assignment, the distrubtion ted to be skewed right for
# most numeric data, which makes sense because they tend to indicate higher socialeconomic class and 
# those statstics tend to have outliers in the high end while skewing right.


# In[104]:


# Preforming K-Means on attribute "age" and "fnlweight"
Points = pd.DataFrame()
Points.loc[:, 0] = inc.loc[:,"age"]
Points.loc[:, 1] = inc.loc[:,"fnlwgt"]
ClusterCentroidGuesses = pd.DataFrame() # creating centroid (k = 3)
ClusterCentroidGuesses.loc[:,0] = [-1, 1, 0]
ClusterCentroidGuesses.loc[:,1] = [2, -2, 0]

Labels, ClusterCentroids = KMeans(Points, ClusterCentroidGuesses)

# Change the plot dimensions
plt.rcParams["figure.figsize"] = [8, 8] # Square
# plt.rcParams["figure.figsize"] = [8, 0.5] # Wide
# plt.rcParams["figure.figsize"] = [0.5, 8] # Tall

# Visualize the results of the K-Means clustering
Title = 'K-Means Test'
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)

# Reset the plot dimensions
plt.rcParams["figure.figsize"] = [6.0, 4.0] # Standard


# In[ ]:




