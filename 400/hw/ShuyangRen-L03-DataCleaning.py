#!/usr/bin/env python
# coding: utf-8

# This is my submission for the lesson 3 assignment. Below are two arrays and 3 functions that will manipulate the dataset. Each functions have comments that explains what they do. At the end of the file are the result of the 3 functions in action (when called). All 3 functions will return an array with the desired modifications. With the exception of function remove_outlier, the functions will return an array with data type float to ensure no information is lost (for mean and medians). All three functions are expected to take in an array of scalars as arguments.

# In[225]:


import numpy as np


# In[250]:


arr1 = np.array([1, 1, 2, 3, 5, 8, 13, 5, 2, 4, 2, 3, 8, 4, 2, 6, 7, 3, 6, 7, 2, 1, 8, 7, 9, 14, 22, 142, 12, 11,]) #set first array to be of at least 30 elements that includes outliers


# In[227]:


arr2 = np.array([2, 1, 3, 6, 2, "?", 3, 5, 6, 4, 8, 11, 7, " ", True]) # second array that includes improper non-numeric missing values


# In[228]:


def remove_outlier(arr): # Will take in Array A of numbers and return Array A with its outliers removed
    LimitHi = np.mean(arr) + 2*np.std(arr) # The high limit for acceptable values is the mean plus 2 standard deviations    
    LimitLo = np.mean(arr) - 2*np.std(arr) # The low limit for acceptable values is the mean plus 2 standard deviations
    FlagGood = (arr >= LimitLo) & (arr <= LimitHi) # Create Flag for values within limits 
    arr = arr[FlagGood] # Overwrite x with the selected values
    return arr # returned the updated array


# In[251]:


def replace_outlier(arr): # Will take in Array A and return Array A with its outliers replaced with the mean of the non-outliers
    LimitHi = np.mean(arr) + 2*np.std(arr) # The high limit for acceptable values is the mean plus 2 standard deviations    
    LimitLo = np.mean(arr) - 2*np.std(arr) # The low limit for acceptable values is the mean plus 2 standard deviations
    FlagGood = (arr >= LimitLo) & (arr <= LimitHi) # Create Flag for values within limits 
    FlagBad = (arr < LimitLo) | (arr > LimitHi) # Create Flag for values out of limits 
    arr = arr.astype(float) #change datatype to float in order to accomadate the mean
    arr[FlagBad] = np.mean(arr[FlagGood]) # Replace outliers with the mean of non-outliers
    return arr # returned the updated array


# In[253]:


def fill_median(arr): # Will take in Array A and return Array A with all miss/non-numeric values changed to the median of the acceptable values
    FlagGood = [element.isdigit() for element in arr] # Find the entries and values that are acceptable
    FlagBad = [not i for i in FlagGood] # mark entries that have missing values that needs to be replaced
    arr[FlagBad] = np.median(arr[FlagGood].astype(float)) # find the median
    arr = arr.astype(float) # setting arr back to an array with datatype float
    return arr # returned the updated array


# In[254]:
print("This is array1:")
print(arr1)
print("This is array2:")
print(arr2)

print("Calling remove_outlier on array1: ")
print(remove_outlier(arr1)) # remove outliers for arr1

arr1 = np.array([1, 1, 2, 3, 5, 8, 13, 5, 2, 4, 2, 3, 8, 4, 2, 6, 7, 3, 6, 7, 2, 1, 8, 7, 9, 14, 22, 142, 12, 11,]) #re-initialize arr1


# In[255]:

print("Calling replace_outlier on array1: ")
print(replace_outlier(arr1)) # replace_outlier for arr1


# In[256]:

print("Calling fill_median on array2: ")
print(fill_median(arr2)) # fill the missing values in arr2 with the median of the acceptable values

