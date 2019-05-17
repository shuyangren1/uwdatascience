#!/usr/bin/env python
# coding: utf-8

# In[224]:


# Import the nessessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

''' The following is a list of Auxiliary functions '''

# Removes outliers that are more than 2 standardd deviantions from the mean
def remove_outliers(x):
    limit_hi = np.mean(x) + 2*np.std(x)
    limit_lo = np.mean(x) - 2*np.std(x)
    flag_good = (x >= limit_lo) & (x <= limit_hi)
    return x[flag_good]

# Normalize numeric values (z-normalization)
def _z_normalize(vals):
    return (vals - np.mean(vals))/np.std(vals)

# function for k means
def _k_means(df, col_1_name, col_2_name, n_clusters):
        X = pd.DataFrame()
        X.loc[:, 0] = _z_normalize(df[col_1_name])
        X.loc[:, 1] = _z_normalize(df[col_2_name])
        kmeans = KMeans(n_clusters)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        plt.scatter(X.loc[:, 0], X.loc[:, 1], c=y_kmeans)
        plt.xlabel(col_1_name)
        plt.ylabel(col_2_name)
        plt.show()
        return y_kmeans

# code to plot roc curve
def _plot_roc(y_test, y_score, classifier):
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="red", lw=2, label='ROC curve (area under curve: %.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for %s' % classifier)
    plt.legend(loc="lower right")
    plt.show()
  
'''' End of Auxiliary Functions'''


# In[225]:


''' Data Preperation'''

# load in dataset from URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" # data set of income/adults
inc = pd.read_csv(url, sep=", ", header=None, engine="python")
# assign reasonable names to columns
inc.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income_above_50K?"]
inc.head()


# In[226]:


print("Number of observations:", len(inc)) # count observation
print("Number of attributes: ", len(inc.columns)) # count attributes
print("Data types: ", inc.dtypes) # show data types


# In[227]:


# Show distribution for numeric values
print("Numeric attribute means: ", inc.mean())
print("Numeric attribute standard deviation: ", inc.std())


# In[228]:


# Show distrubtion of categorical attributes
for col in inc.columns:
    if type(inc[col][1]) == str:
        inc[col].value_counts().plot(kind='bar')
        plt.show()


# In[229]:


# Comment for each attribute:
# workclass            categorical attribute of the person's workclass
# fnlwgt               a numeric value that controlled to independent estimates of the civilian noninstitutional population of the US
#                      for info can be found here: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
# education            categorical attribute of the highest level of education the person has aquired
# education-num        numeric value of the number of years of education the person has
# marital-status       categorical attribute of the person's marital status
# occupation           categorical attribute of the person's work occupation
# relationship         categorical attribute of person's relationship to family (husband, wife, etc)
# race                 categorical attribute of person's race/ethnicity
# sex                  categorical attribute of the person's sex
# capital-gain         numeric attribute of person's capital gain
# capital-loss         numeric attribute of person's capital loss
# hours-per-week       numeric attribute of person's average hours worked per week
# native-country       categorical attribute of person's country of origin
# income_above_50K?    categorical attribute of whether the person's income is above 50K a year


# In[230]:


# Removing missing values
# This data set is actually has very little missing data, and of the missing ones we don't want to remove
# the entire row because it would also affect workclassSo we will just do this on one column as an example
x = inc.loc[:, "sex"] != "?"
inc.loc["age"] = x


# In[231]:


# Removing outliers
# We are only doing this on hours-per-week for the purpose of this excerise
# because we are interested in the outliers of other attributes
# in economics, income is natrually very unequal and outliers are very common. We don't want to remove them.
# But below is an example of what the code would look like if we were to remove outliers for hours-per-week
# inc.loc[:,"hours-per-week"] = remove_outliers(inc.loc[:,"hours-per-week"])


# In[232]:


inc.dtypes


# In[233]:


# Impute missing values for hours-per-week
median = np.nanmedian(inc.loc[:,"hours-per-week"]) # Replace missing values with the median of nonmissing values
inc.loc[:,"hours-per-week"] = inc.loc[:,"hours-per-week"].fillna(median)
pd.unique(inc.loc[:,'hours-per-week']) # check if any NaN is still left
# Repeat for other attributes


# In[234]:


median = np.nanmedian(inc.loc[:,"age"]) 
inc.loc[:,"age"] = inc.loc[:,"age"].fillna(median)

median = np.nanmedian(inc.loc[:,"fnlwgt"]) 
inc.loc[:,"fnlwgt"] = inc.loc[:,"fnlwgt"].fillna(median)

median = np.nanmedian(inc.loc[:,"education-num"])
inc.loc[:,"education-num"] = inc.loc[:,"education-num"].fillna(median)

median = np.nanmedian(inc.loc[:,"capital-gain"]) 
inc.loc[:,"capital-gain"] = inc.loc[:,"capital-gain"].fillna(median)

median = np.nanmedian(inc.loc[:,"capital-loss"])
inc.loc[:,"capital-loss"] = inc.loc[:,"capital-loss"].fillna(median)


# In[235]:


# Consolidate categorical data  for education
# First show the different categories in education
inc.loc[:,"education"].unique()


# In[236]:


# Consolidate education into "No-HS", "Some-HS", "HS-Grad", "Some-College", "Associates", "Bachelors", "Masters", "Doctorate", and "Prof-school"
inc.loc[inc.loc[:, "education"].isin(["Preschool", "1st-4th", "5th-6th", "7th-8th"]), "education"] = "No-HS"
inc.loc[inc.loc[:, "education"].isin(["9th", "10th", "11th", "12th"]), "education"] = "Some-HS"
inc.loc[inc.loc[:, "education"].isin(["Assoc-acdm", "Assoc-voc"]), "education"] = "Associates"
pd.value_counts(inc.loc[:,"education"]) #let's take another look at the column education


# In[237]:


# One-hot encode categorical data
inc.loc[:, "HS-grad"] = (inc.loc[:, "education"] == "HS-grad").astype(int)
inc.loc[:, "Some-college"] = (inc.loc[:, "education"] == "Some-college").astype(int)
inc.loc[:, "Bachelors"] = (inc.loc[:, "education"] == "Bachelors").astype(int)
inc.loc[:, "Some-HS"] = (inc.loc[:, "education"] == " Some-HS").astype(int)
inc.loc[:, "Associates"] = (inc.loc[:, "education"] == "Associates").astype(int)
inc.loc[:, "Masters"] = (inc.loc[:, "education"] == "Masters").astype(int)
inc.loc[:, "No-HS"] = (inc.loc[:, "education"] == "No-HS").astype(int)
inc.loc[:, "Prof-School"] = (inc.loc[:, "education"] == "Prof-school").astype(int)
inc.loc[:, "Doctorate"] = (inc.loc[:, "education"] == "Doctorate").astype(int)
inc.loc[:, "married"] = inc.loc[:, "relationship"].map({"Not-in-family":0, "Unmarried":0, "Own-child":0, "Other-relative":0, "Husband":1, "Wife":1}).fillna(-1).astype(int)
inc.loc[:, "encoded_workclass"] = inc.loc[:, "workclass"].map({"Private": 0, "State-gov": 1, "Federal-gov": 2, "Self-emp-not-inc": 3, "Self-emp-inc": 4, "Local-gov": 5, "Without-pay": 6}).fillna(-1).astype(int)
inc.loc[:, "encoded_occupation"] = inc.loc[:, "occupation"].map({"Priv-house-serv": 0, "Protective-serv": 1, "Handlers-cleaners": 2, "Machine-op-inspct": 3, "Adm-clerical": 4, "Farming-fishing": 5, "Transport-moving": 6, "Craft-repair": 7, "Other-service": 8, "Tech-support": 9, "Sales": 10, "Exec-managerial": 11, "Prof-specialty": 12, "Armed-Forces": 13}).fillna(-1).astype(int)
inc.loc[:, "from_US"] = np.where(inc.loc[:, "native-country"] == "United-States", 0, 1)
inc.loc[:, "income_above_50K"] = np.where(inc.loc[:, "income_above_50K?"] == "<=50K", 0, 1)
inc.head()


# In[238]:


# Normalization - Done within K-means (see below)


# In[239]:


'''Unsupervised learning'''

# Performing K-means. See the axuiliary function: normalization is done within _k_means
# Includes one categorical variable (occupation) and one numeric attribute (hours-per-week)
y_kmeans = _k_means(inc, "hours-per-week", "encoded_occupation", 2)


# In[240]:


# adding cluster label to the data 
inc.loc[:, "cluster"] = y_kmeans
inc.head()


# In[241]:


inc.columns


# In[242]:


''' Supervised Learning'''

# The Binary question we are going to ask: do marital status serve as a good predictor for whether or not a person makes over 50K a year?
# Splitting data
X = inc[["age", "education-num", "capital-gain", "capital-loss", "hours-per-week", 'HS-grad', 'Some-college', 'Bachelors', 'Some-HS','Associates', 'Masters', 'No-HS', 'Prof-School', 'Doctorate',  "married", "encoded_workclass", "from_US", "cluster"]]
y = inc["income_above_50K"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[243]:


# Using sklearn to train 2 classifiers
# Using Logistic Regression
log_clf = LogisticRegression()
log_clf.fit(x_train, y_train)
# Using Decision Tree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train, y_train)


# In[244]:


estimators = 10 # number of trees parameter
mss = 2 # mininum samples split parameter
rf_clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss)
rf_clf.fit(x_train, y_train)


# In[245]:


# Applying the trained classifiers
log_pred = log_clf.predict(x_test)
dt_pred = dt_clf.predict(x_test)
rf_pred = rf_clf.predict(x_test)


# In[246]:


# Constructing a confusion matrix for each classifier    

baseline = pd.DataFrame(y_test)
baseline["income_above_50K"] = 0
tn, fp, fn, tp = confusion_matrix(y_test, baseline).ravel()
threshold = (tp + tn)/(tp + tn + fp + fn)
print("Threashold: ", threshold)


# In[247]:


log_k = pd.DataFrame(confusion_matrix(y_test, log_pred))
print("Logistic Regression confusion matrix: \n", log_k)
dt_k = pd.DataFrame(confusion_matrix(y_test, dt_pred))
print("Decision Tree confusion matrix: \n", dt_k)
rf_k = pd.DataFrame(confusion_matrix(y_test, rf_pred))
print("Random Forest confusion matrix: \n", rf_k)


# In[248]:


# Accuracy metrics for each classifer based on the confusion matrix
print("Logistic Regression: \n", classification_report(log_pred, y_test, target_names=["income <= 50k", "income > 50k"]))
print("Decision Tree: \n", classification_report(log_pred, y_test, target_names=["income <= 50k", "income > 50k"]))
print("Random Forest: \n", classification_report(rf_pred, y_test, target_names=["income <= 50k", "income > 50k"]))


# In[249]:


# Calculating ROC curve and AUC using sklearn for all classifer
_plot_roc(y_test, log_clf.fit(x_train, y_train).decision_function(x_test), "Logistic Regression")
_plot_roc(y_test, dt_clf.fit(x_train, y_train).predict_proba(x_test)[:,1], "Decision Tree")
_plot_roc(y_test, rf_clf.fit(x_train, y_train).predict_proba(x_test)[:,1], "Random Forest")

