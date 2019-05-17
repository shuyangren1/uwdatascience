#!/usr/bin/env python
# coding: utf-8

# For my Milestone 2, my data set is an income census of adults. Specifically, it surveys the participants on numeric qualities such as age, years of education, and capital gain, as well as qualitative attributes such as marital status, occupation, and working class.
# Additionally,the focus of the survey is whether or not the participant has an annual income above $50,000, which also acts as a binary column.

# In[164]:


# Import statements for necessary package(s).
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
# Loading in dataset from internet
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" # data set of income/adults
inc = pd.read_csv(url, sep=", ", header=None, engine="python")
# assign reasonable names to columns
inc.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income_above_50K?"]
inc.head()
##############


# In[182]:


np.unique(inc.loc[:,'race'])


# In[210]:


# Data preperation - setting up functions

# normalize takes in an array of values and return that array normalized (z-nomalization)
def normalize(x):
    offset = np.mean(x) # offset is the mean
    spread = np.std(x) # spread is the standard deviation
    return (x - offset) / spread # return x after it is z-normalized

# this function will take in a categorical column, and return the encoded columns in a list
def one_hot_encode(x):
    unique_values = np.unique(x)
    result = []
    for element in unique_values:
        new_column = (x == element)
        result.append(new_column)
    return result

# takes in an numeric array and returns the array with its missing values ("?" and "NA") replaced with the array median
def replace_missing(x):
    x = pd.replace(to_replace="NA", value=float("NaN")) # Replace "NA" values as NaN
    x = pd.replace(to_replace="?", value=float("NaN")) # Replace "?" values as NaN
    median = np.nanmedian(x) # set the median for which is to replace NaN values
    for e in x: # Iterate through x to replace all nan values with the median
        if np.isnan(e):
            e = median

def replace_outlier(arr): # Will take in Array A and return Array A with its outliers replaced with the mean of the non-outliers
    LimitHi = np.mean(arr) + 2*np.std(arr) # The high limit for acceptable values is the mean plus 2 standard deviations    
    LimitLo = np.mean(arr) - 2*np.std(arr) # The low limit for acceptable values is the mean plus 2 standard deviations
    FlagGood = (arr >= LimitLo) & (arr <= LimitHi) # Create Flag for values within limits 
    FlagBad = (arr < LimitLo) | (arr > LimitHi) # Create Flag for values out of limits 
    arr = arr.astype(float) #change datatype to float in order to accomadate the mean
    arr[FlagBad] = np.mean(arr[FlagGood]) # Replace outliers with the mean of non-outliers
    return arr # returned the updated array


# In[216]:


a = np.array([1,2,3,4,100])
replace_outlier(a)


# In[219]:


# Normalize some numeric columns
inc["norm_age"] = normalize(inc.loc[:,"age"])
inc["norm_fnlwgt"] = normalize(inc.loc[:,"norm_fnlwgt"])
inc["norm_education-num"] = normalize(inc.loc[:,"education-num"])
inc["norm_hours-per-week"] = normalize(inc.loc[:,"hours-per-week"])

# One-hot encode categorical column
# Remove and replace missing values
# Remove and replace outliers.


# In[156]:


# Ask a binary-choice question that describes your classification. Write the question as a comment. Specify an appropriate column as your expert label for a classification (include decision comments).


# In[157]:


# Apply K-Means on some of your columns, but make sure you do not use the expert label. Add the K-Means cluster labels to your dataset.


# In[158]:


# Split your data set into training and testing sets using the proper function in sklearn (include decision comments).


# In[159]:


# Create a classification model for the expert label based on the training data (include decision comments).


# In[160]:


# Apply your (trained) classifiers to the test data to predict probabilities.


# In[161]:


# Write out to a csv a dataframe of the test data, including actual outcomes, and the probabilities of your classification


# In[163]:


# Determine accuracy rate, which is the number of correct predictions divided by the total number of predictions (include brief preliminary analysis commentary).


# Summary:

# In[ ]:




