#!/usr/bin/env python
# coding: utf-8

# In[142]:


#!/usr/bin/env python
# coding: utf-8

# This is my submission for assignment of lesson 4. I have used a dataset of cars with attribute relating to car characteristics such as
# miles per hour, model year, weight, etc. This is similar to what was used in the lab but I am using the version that was unedited (so it has missing values).
# For the majority of these tasks I am focusing on MPG. These codes can be replicated for the other attributes by simply replacing "MPG" with the new attribute name


#Import statements for necessary package(s).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
#Read in the dataset 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original" # data set of cars, our focus is on mpg efficency
cars = pd.read_csv(url,sep='\s+', header=None)
cars.head()


# In[143]:


cars.columns


# In[144]:


#Assign reasonable column names
cars.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
cars.head()


# In[145]:


#Impute and assign median values for missing numeric values.
cars.replace(to_replace="NA", value=float("NaN")) #replace "NA" values as NaN
HasNan = np.isnan(cars.loc[:,"mpg"]) #mark values that is NaN
cars.loc[HasNan, "mpg"] = np.nanmedian(cars.loc[:,"mpg"]) #replace NaN values with the median
# check how many missing values
cars.isnull().sum()


# In[146]:


#Replace outliers for mpg
#first set the upper and lower limites for determining outliers
LimitHi = np.mean(cars.loc[:, "mpg"]) + 2*np.std(cars.loc[:,"mpg"])
LimitLo = np.mean(cars.loc[:, "mpg"]) - 2*np.std(cars.loc[:,"mpg"])
#then flag the values that needs to be replaced and values that don't need to be
FlagGood = (cars.loc[:, "mpg"] >= LimitLo) & (cars.loc[:, "mpg"] <= LimitHi)
FlagBad = (cars.loc[:, "mpg"] < LimitLo) | (cars.loc[:, "mpg"] > LimitHi)
cars.loc[FlagBad, "mpg"] = np.mean(cars.loc[FlagGood, "mpg"]) # Replace outliers with the mean of non-outliers


# In[147]:


sum(FlagBad) #count the total number of outliers


# In[148]:


# Repeat for the other interested numeric values
# For displacement
LimitHi = np.mean(cars.loc[:, "displacement"]) + 2*np.std(cars.loc[:,"displacement"])
LimitLo = np.mean(cars.loc[:, "displacement"]) - 2*np.std(cars.loc[:,"displacement"])
FlagGood = (cars.loc[:, "displacement"] >= LimitLo) & (cars.loc[:, "displacement"] <= LimitHi)
FlagBad = (cars.loc[:, "displacement"] < LimitLo) | (cars.loc[:, "displacement"] > LimitHi)
cars.loc[FlagBad, "displacement"] = np.mean(cars.loc[FlagGood, "displacement"])
print("Number of outliers for displacment: ", sum(FlagBad))
# For horsepower
LimitHi = np.mean(cars.loc[:, "horsepower"]) + 2*np.std(cars.loc[:,"horsepower"])
LimitLo = np.mean(cars.loc[:, "horsepower"]) - 2*np.std(cars.loc[:,"horsepower"])
FlagGood = (cars.loc[:, "horsepower"] >= LimitLo) & (cars.loc[:, "horsepower"] <= LimitHi)
FlagBad = (cars.loc[:, "horsepower"] < LimitLo) | (cars.loc[:, "horsepower"] > LimitHi)
cars.loc[FlagBad, "horsepower"] = np.mean(cars.loc[FlagGood, "horsepower"])
print("Number of outliers for horsepower: ", sum(FlagBad))
# For weight
LimitHi = np.mean(cars.loc[:, "weight"]) + 2*np.std(cars.loc[:,"weight"])
LimitLo = np.mean(cars.loc[:, "weight"]) - 2*np.std(cars.loc[:,"weight"])
FlagGood = (cars.loc[:, "weight"] >= LimitLo) & (cars.loc[:, "weight"] <= LimitHi)
FlagBad = (cars.loc[:, "weight"] < LimitLo) | (cars.loc[:, "weight"] > LimitHi)
cars.loc[FlagBad, "weight"] = np.mean(cars.loc[FlagGood, "weight"])
print("Number of outliers for weight: ", sum(FlagBad))
# For acceleration
LimitHi = np.mean(cars.loc[:, "acceleration"]) + 2*np.std(cars.loc[:,"acceleration"])
LimitLo = np.mean(cars.loc[:, "acceleration"]) - 2*np.std(cars.loc[:,"acceleration"])
FlagGood = (cars.loc[:, "acceleration"] >= LimitLo) & (cars.loc[:, "acceleration"] <= LimitHi)
FlagBad = (cars.loc[:, "acceleration"] < LimitLo) | (cars.loc[:, "acceleration"] > LimitHi)
cars.loc[FlagBad, "acceleration"] = np.mean(cars.loc[FlagGood, "acceleration"])
print("Number of outliers for acceleration: ", sum(FlagBad))


# In[149]:


#Create a histogram of a numeric variable. Use plt.show() after each histogram
plt.hist(cars.loc[:,'mpg'])
plt.show()


# In[150]:


# Creating a scatterplot. Using plt.show() after the scatterplot.
# A scatter plot of car's model year on the x-axis and mpg on the y-axis. As expected over time mpg seems to rises
plt.scatter(cars.loc[:,'model_year'], cars.loc[:,'mpg'])
plt.show()


# In[151]:


#check the attribute's datatype
cars.dtypes


# In[152]:


# The standard deviation of all numeric variables. Using print() for each standard deviation.
print(np.std(cars.loc[:,"mpg"]))
print(np.std(cars.loc[:,"cylinders"]))
print(np.std(cars.loc[:,"displacement"]))
print(np.std(cars.loc[:,"horsepower"]))
print(np.std(cars.loc[:,"weight"]))
print(np.std(cars.loc[:,"acceleration"]))
print(np.std(cars.loc[:,"model_year"]))
print(np.std(cars.loc[:,"origin"]))


# Answers for Q10.
# - Which attributes had outliers and how were the outliers defined?
# Outliers are defined as values that are more than 2 standard deviations away from the mean. All of the interested numeric variables had outliers: mpg, displacment, horsepower, weight, and acceleration
# - Which attributes required imputation of missing values and why?
# Only horsepower contained missing value because when we called "cars.isnull().sum(), only horsepower had non-zero count for null values
# - Which attributes were histogrammed and why?
# displacement, horsepower, weight,accleration, and mpg were historgrammed because they were numeric values of the most interest. There are a few other numeric values but they either not as relevant are more categorical-like (e.g. number of cylinders for the engine, which is more categorical in regard to engine type).
# - Which attributes were removed and why?
# I would remove origin, because there was really not any information given on what the attribute means.
# - How did you determine which rows should be removed?
# I did not remove any rows, but if I would I only would remove if it is missing values. But the only value missing is horsepower for 6 entries and I didn't feel like that was enough reason to remove them.

# In[ ]:




