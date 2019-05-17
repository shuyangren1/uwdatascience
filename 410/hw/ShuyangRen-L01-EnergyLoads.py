#!/usr/bin/env python
# coding: utf-8

# # Lesson 1 Assignment - EnergyLoads
# ## Author - Studentname

# ### Instructions
# In this exercise you will use visualization and summary statistical methods to explore a data set. You will be looking for interesting aspects in these data. Specifically, you will explore an energy efficiency data set.
# 
# This data set contains characteristics of over 750 buildings. The efficiency of the building is measured as either heating load or cooling load. Your goal is to find **three interesting relationships within the variables** which help you understand the energy efficiency of these buildings.
# 
# The exercise is deliberately open-ended. Whenever you approach a new data set some open-ended exploration is required. Expect this exploration to be an iterative process. You may need to try several ideas before you find truly interesting relationships.
# 
# The key points come from examining charts and summary statistics based on distribution Heating Load and Cooling Load. 
# <font color="blue">At the end of this notebook include Markdown cells outlining your 3 key points.</font>
# 
# Example description:  The heating load of buildings depends on ... Evidence for this relationship can be seen by ... in the figure and by noting .... in the table above. 
# 

# #### Tip: 
# There are no categorical variables in this data set. There are two numeric variables, Orientation and Glazing Area Distribution, with only 4 and 2 discrete values. You can convert these integer values to factors and examine the effect on the distribution of Heating Load and Cooling Load.

# In[1]:


# Import libraries
import pandas as pd
import numpy as np


# In[2]:


# Cleaned up Dataset location
fileName = "https://library.startlearninglabs.uw.edu/DATASCI410/Datasets/EnergyEfficiencyData.csv"


# In[3]:


# read in dataset
data = pd.read_csv(fileName)


# In[4]:


# take a quick look at the data
data.head()


# In[5]:


# check for null values
data.isnull().values.any()


# In[6]:


# replace pesky spaces in column names
data.columns = data.columns.str.replace(' ', '_')


# In[7]:


# check that it worked
data.head()


# In[8]:


# view dimensions
data.shape


# In[9]:


# view datatypes
data.dtypes  # yep no categorical variables


# In[10]:


# look at summary stats
data.Surface_Area.describe()


# In[11]:


# view variable distributions
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(6, 6)) # Define plot area
ax = fig.gca() # Define axis 
data.boxplot()
plt.xticks(rotation=90) # move to vertical axes so we can see labels
ax.set_title('Box plot of variables') # Give the plot a main title
plt.show()


# In[12]:


data.columns


# In[13]:


data.dtypes


# In[15]:


#convert Orientation and Glazing Area Distribution to factors to examine their relationship to heating and cooling
data["Orientation"] = data.Orientation.astype("category")
data["Glazing_Area_Distribution"] = data.Glazing_Area_Distribution.astype("category")


# In[16]:


data.head()


# In[17]:


# using pandas corr() function to create a correlation matrix for our dataset
corr = data.corr()
print(corr)


# In[18]:


#using graphs to visualize the data
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.xticks(rotation=90)
plt.show()


# In[19]:


# using seaborn to get a legend that helps us to identify the relationships
import seaborn as sns
corr = data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[20]:


# using the cov() function to view covariance within the data.
data.cov()


# In[21]:


data[['Surface_Area','Heating_Load']].corr()


# In[22]:


data[['Surface_Area','Heating_Load']].cov()


# Three intersting relations I found:
# - roof area is inversely correlated with overall height
# - surface area is inversely correlated with overall height
# - surface area is inversely correlated with relative compactness

# In[ ]:




