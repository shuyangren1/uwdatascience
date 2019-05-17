#!/usr/bin/env python
# coding: utf-8

# This is my submission for the Milestone 2 assignment, where I implemented code for a data set reporting on income/census.
# Each of the tasks listed in the assignment is seperated by "##############".
# The data set contains more than 1 column of numeric values. However, due to the nature of this data, we are very much interested in the outliers. So, for tasks such as "replace all outliers in numeric value columns", I have decided to only implemement the code for one such column ("hours-per-week") to perserve the outliers. To replace outliers of other columns, simply repeat the code I have for "hours-per-week" and change all instances of "hours-per-week" to the other columns' name". Similarly, the same is true for normalization of numeric variables where you take the code I have for "age" and repeat it for the other column's numeric data by replacing all instances of "age" with that column's name.
# Another thing worth noting is that this data set has no missing numeric values. So, for the task where missing numeric variables were to be replaced, I simply implemented the code that would have worked if there was really missing numerical data for one of the columns (see below).

# In[72]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
#load in dataset from URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" # data set of income/adults
inc = pd.read_csv(url, sep=", ", header=None, engine="python")
# assign reasonable names to columns
inc.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income_above_50K?"]
inc.head()
##############


# In[41]:


inc.dtypes # Show datatypes


# In[73]:


# Replace outliers for numeric column "hours-per-week"
# First set the upper and lower limites for determining outliers
LimitHi = np.mean(inc.loc[:, "hours-per-week"]) + 2*np.std(inc.loc[:,"hours-per-week"])
LimitLo = np.mean(inc.loc[:, "hours-per-week"]) - 2*np.std(inc.loc[:,"hours-per-week"])
#then flag the values that needs to be replaced and values that don't need to be
FlagGood = (inc.loc[:, "hours-per-week"] >= LimitLo) & (inc.loc[:, "hours-per-week"] <= LimitHi)
FlagBad = (inc.loc[:, "hours-per-week"] < LimitLo) | (inc.loc[:, "hours-per-week"] > LimitHi)
# Replace outliers with the mean of non-outliers
inc.loc[FlagBad, "hours-per-week"] = np.mean(inc.loc[FlagGood, "hours-per-week"])
##############


# In[74]:


# Impute and assign median values for missing numeric values.
# In actuality, the dataset is not missing any numeric data.
# But for the sake of the assignment, I will implement the code for replacing missing numeric data in "fnlwgt"
inc = inc.replace(to_replace="NA", value=float("NaN")) #replace "NA" values as NaN
inc = inc.replace(to_replace="?", value=float("NaN")) #replace "?" values as NaN
HasNan = np.isnan(inc.loc[:,"fnlwgt"]) #mark values that is NaN
inc.loc[HasNan, "fnlwgt"] = np.nanmedian(inc.loc[:,"fnlwgt"]) #replace NaN values with the median
# Check how many missing values
inc.isnull().sum()
# As you can see, the only missing values are in categorical columns
##############


# In[75]:


# Normalizing the numeric values for "age"
inc.loc[:, "age"] = pd.to_numeric(inc.loc[:, "age"], errors='coerce') #coercing values to int/float
offset = np.mean(inc.loc[:, "age"]) #offset is the mean
spread = np.std(inc.loc[:, "age"])#spread is the standard deviation
inc.loc[:, "age_znorm"] = (inc.loc[:, "age"] - offset)/spread #normalizing age and saving it to a new column
plt.hist(inc.loc[:, "age_znorm"]) #displaying the historgram for the distribution
plt.show()
print(inc.head())
##############


# In[76]:


# Bin numeric variable age (equal-width)
# Determine the boundaries of the bins
NumberOfBins = 3
BinWidth = (max(inc.loc[:, "age"]) - min(inc.loc[:, "age"]))/NumberOfBins
MinBin1 = float('-inf')
MaxBin1 = min(inc.loc[:, "age"]) + 1 * BinWidth
MaxBin2 = min(inc.loc[:, "age"]) + 2 * BinWidth
MaxBin3 = float('inf')
print("Bin 1 is greater than", MinBin1, "up to", MaxBin1)
print("Bin 2 is greater than", MaxBin1, "up to", MaxBin2)
print("Bin 3 is greater than", MaxBin2, "up to", MaxBin3)
# Create the categorical variable
# Start with an empty array that is the same size as age
binnedAge = np.empty(len(inc.loc[:, "age"]), object)
binnedAge[(inc.loc[:, "age"] > MinBin1) & (inc.loc[:, "age"] <= MaxBin1)] = "Young"
binnedAge[(inc.loc[:, "age"] > MaxBin1) & (inc.loc[:, "age"] <= MaxBin2)] = "Middle-Aged"
binnedAge[(inc.loc[:, "age"] > MaxBin2) & (inc.loc[:, "age"] <= MaxBin3)] = "Old"
print("Age is binned into 3 equal-width bins:", binnedAge)
inc.loc[:, "Binned_Age"] = binnedAge #save the array to the table
##############


# In[59]:


# Consolidate categorical data  for education
# First show the different categories in education
inc.loc[:,"education"].unique()


# In[77]:


# Consolidate education into "No-HS", "Some-HS", "HS-Grad", "Some-College", "Associates", "Bachelors", "Masters", "Doctorate", and "Prof-school"
inc.loc[inc.loc[:, "education"].isin(["Preschool", "1st-4th", "5th-6th", "7th-8th"]), "education"] = "No-HS"
inc.loc[inc.loc[:, "education"].isin(["9th", "10th", "11th", "12th"]), "education"] = "Some-HS"
inc.loc[inc.loc[:, "education"].isin(["Assoc-acdm", "Assoc-voc"]), "education"] = "Associates"
pd.value_counts(inc.loc[:,"education"]) #let's take another look at the column education
##############


# In[66]:


# One-hot encode categorical data for Horsepower "income_above_50K?"
# First see how many categories are there for this variable
inc.loc[:,"income_above_50K?"].unique()


# In[78]:


# Create 2 new columns, one for each state in "income_above_50K?"
inc.loc[:, "Income <= 50K"] = (inc.loc[:, "income_above_50K?"] == "<=50K").astype(int)
inc.loc[:, "Income > 50K"] = (inc.loc[:, "income_above_50K?"] == ">50K").astype(int)
inc.head()
##############


# In[79]:


# Remove obsolete column "horsepower" now that we have broken it down into dummy variables
inc = inc.drop("income_above_50K?", axis=1)
inc.head()
##############


# In[ ]:




