#!/usr/bin/env python
# coding: utf-8

# This is my submission for Lesson 5 Assignment. I imported the same dataset of cars as I had used for Lesson 4 Assignment. The numberical variable of focus here is mpg, which will be used for normalization. Horsepower is numerical but will be transformed into categorical for this excerise by splitting the observation into "Low", "Medium", and "High" Horsepower categories. Cylinders is another variable, while seemingly numeric, can technically be interrupted as categroical as it is defining the engine type. These are the variables of focus that will be used in the code below.

# In[395]:


#import the nessessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
#load in dataset from URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original" # data set of cars, our focus is on mpg efficency
cars = pd.read_csv(url,sep='\s+', header=None)
# assign reasonable names
cars.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
cars.head()


# In[396]:


#normalizing the numeric values for mpg
cars.loc[:, "mpg"] = pd.to_numeric(cars.loc[:, "mpg"], errors='coerce') #coercing values to int/float
offset = np.mean(cars.loc[:, "mpg"]) #offset is the mean
spread = np.std(cars.loc[:, "mpg"])#spread is the standard deviation
mpgNorm = (cars.loc[:, "mpg"] - offset)/spread #normalizing mpg
plt.hist(mpgNorm)
plt.show()


# In[397]:


# Bin numeric variable horsepower (equal width)
# Determine the boundaries of the bins
NumberOfBins = 3
BinWidth = (max(cars.loc[:, "horsepower"]) - min(cars.loc[:, "horsepower"]))/NumberOfBins
MinBin1 = float('-inf')
MaxBin1 = min(cars.loc[:, "horsepower"]) + 1 * BinWidth
MaxBin2 = min(cars.loc[:, "horsepower"]) + 2 * BinWidth
MaxBin3 = float('inf')
print(" Bin 1 is greater than", MinBin1, "up to", MaxBin1)
print(" Bin 2 is greater than", MaxBin1, "up to", MaxBin2)
print(" Bin 3 is greater than", MaxBin2, "up to", MaxBin3)
# Create the categorical variable
# Start with an empty array that is the same size as horsepower
binnedHorsepower = np.empty(len(cars.loc[:, "horsepower"]), object)
binnedHorsepower[(cars.loc[:, "horsepower"] > MinBin1) & (cars.loc[:, "horsepower"] <= MaxBin1)] = "Low"
binnedHorsepower[(cars.loc[:, "horsepower"] > MaxBin1) & (cars.loc[:, "horsepower"] <= MaxBin2)] = "Med"
binnedHorsepower[(cars.loc[:, "horsepower"] > MaxBin2) & (cars.loc[:, "horsepower"] <= MaxBin3)] = "High"
print(" Horsepower binned into 3 equal-width bins:", binnedHorsepower)
############


# In[398]:


# Decode categorical data - cylinders (number of clinders of the engine)
# First check for unique values in cylinders column
cars.loc[:,"cylinders"].unique()


# In[399]:


# Then assign new names for the values
Replace = cars.loc[:, "cylinders"] == 8
cars.loc[Replace, "cylinders"] = "8-Cylinder Engine"

Replace = cars.loc[:, "cylinders"] == 4
cars.loc[Replace, "cylinders"] = "4-Cylinder Engine"

Replace = cars.loc[:, "cylinders"] == 6
cars.loc[Replace, "cylinders"] = "6-Cylinder Engine"

Replace = cars.loc[:, "cylinders"] == 3
cars.loc[Replace, "cylinders"] = "3-Cylinder Engine"

Replace = cars.loc[:, "cylinders"] == 5
cars.loc[Replace, "cylinders"] = "5-Cylinder Engine"

# Get the counts for each value
cars.loc[:,"cylinders"].value_counts()


# In[400]:


# Impute missing categories (for this one, we are converting Horsepower to a categorical data using binning categories)
# First convert horsepower to a categorical data
cars.loc[:,'horsepower'] = binnedHorsepower
# Check for missing values
print("Missing values in horsepower: ",cars.loc[:,"horsepower"].isnull().sum())


# In[401]:


# We see six missing values from horsepower.
# Specify all the locations that have a missing value
MissingValue = (cars.loc[:, "horsepower"]).isnull()
# Impute missing values
cars.loc[MissingValue, "horsepower"] = "Missing"
# Get the counts for each value
cars.loc[:, "horsepower"].value_counts()


# In[402]:


# Consolidate categorical data  for clyinders
# Group together 3-cylinder engines and 5-cylinder engines as odd-cylinder engines
cars.loc[cars.loc[:, "cylinders"].isin(["3-Cylinder Engine", "5-Cylinder Engine"]), "cylinders"] = "Odd-Count Cylinders"
pd.value_counts(cars.loc[:,"cylinders"])


# In[403]:


# One-hot encode categorical data for Horsepower
# Horsepower is already binned and turned into categorical data of "Low", "Med", and "High"

# Create 3 new columns, one for each state in "Margin"
cars.loc[:, "Low Horsepower"] = (cars.loc[:, "horsepower"] == "Low").astype(int)
cars.loc[:, "Medium Horsepower"] = (cars.loc[:, "horsepower"] == "Med").astype(int)
cars.loc[:, "High Horsepower"] = (cars.loc[:, "horsepower"] == "High").astype(int)

# Remove obsolete column "horsepower" now that we have broken it down into dummy variables
cars = cars.drop("horsepower", axis=1)

# Check the first rows of the data frame
cars.head()


# In[415]:


# Plot for categorical columns cylinders
cars["cylinders"].value_counts().plot(kind='bar')
plt.show()


# In[ ]:




