#!/usr/bin/env python
# coding: utf-8

# In[264]:


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


# In[265]:


# load in dataset from URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" # data set of income/adults
inc = pd.read_csv(url, sep=", ", header=None, engine="python")
# assign reasonable names to columns
inc.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income_above_50K?"]


# In[266]:


# Impute missing values for hours-per-week
median = np.nanmedian(inc.loc[:,"hours-per-week"]) # Replace missing values with the median of nonmissing values
inc.loc[:,"hours-per-week"] = inc.loc[:,"hours-per-week"].fillna(median)
pd.unique(inc.loc[:,'hours-per-week']) # check if any NaN is still left
# Repeat for other attributes

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


# Consolidate categorical data  for education
# First show the different categories in education
inc.loc[:,"education"].unique()


# Consolidate education into "No-HS", "Some-HS", "HS-Grad", "Some-College", "Associates", "Bachelors", "Masters", "Doctorate", and "Prof-school"
inc.loc[inc.loc[:, "education"].isin(["Preschool", "1st-4th", "5th-6th", "7th-8th"]), "education"] = "No-HS"
inc.loc[inc.loc[:, "education"].isin(["9th", "10th", "11th", "12th"]), "education"] = "Some-HS"
inc.loc[inc.loc[:, "education"].isin(["Assoc-acdm", "Assoc-voc"]), "education"] = "Associates"
pd.value_counts(inc.loc[:,"education"]) #let's take another look at the column education


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


# In[267]:


'''Unsupervised learning'''

# Performing K-means. See the axuiliary function: normalization is done within _k_means
# Includes one categorical variable (occupation) and one numeric attribute (hours-per-week)
y_kmeans = _k_means(inc, "hours-per-week", "encoded_occupation", 2)


# In[240]:


# adding cluster label to the data 
inc.loc[:, "cluster"] = y_kmeans
inc.head()


# In[268]:


''' Supervised Learning'''

# The Binary question we are going to ask: do marital status serve as a good predictor for whether or not a person makes over 50K a year?
# Splitting data
X = inc[["age", "education-num", "capital-gain", "capital-loss", "hours-per-week", 'HS-grad', 'Some-college', 'Bachelors', 'Some-HS','Associates', 'Masters', 'No-HS', 'Prof-School', 'Doctorate',  "married", "encoded_workclass", "from_US", "cluster"]]
y = inc["income_above_50K"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[269]:


# Using sklearn to train 2 classifiers
# Using Logistic Regression
log_clf = LogisticRegression()
log_clf.fit(x_train, y_train)
# Using Decision Tree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train, y_train)


# In[270]:


estimators = 10 # number of trees parameter
mss = 2 # mininum samples split parameter
rf_clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss)
rf_clf.fit(x_train, y_train)


# In[271]:


# Applying the trained classifiers
log_pred = log_clf.predict(x_test)
dt_pred = dt_clf.predict(x_test)
rf_pred = rf_clf.predict(x_test)


# In[272]:


# Constructing a confusion matrix for each classifier    

baseline = pd.DataFrame(y_test)
baseline["income_above_50K"] = 0
tn, fp, fn, tp = confusion_matrix(y_test, baseline).ravel()
threshold = (tp + tn)/(tp + tn + fp + fn)
print("Threashold: ", threshold)


# In[273]:


log_k = pd.DataFrame(confusion_matrix(y_test, log_pred))
print("Logistic Regression confusion matrix: \n", log_k)
dt_k = pd.DataFrame(confusion_matrix(y_test, dt_pred))
print("Decision Tree confusion matrix: \n", dt_k)
rf_k = pd.DataFrame(confusion_matrix(y_test, rf_pred))
print("Random Forest confusion matrix: \n", rf_k)


# In[274]:


# Accuracy metrics for each classifer based on the confusion matrix
print("Logistic Regression: \n", classification_report(log_pred, y_test, target_names=["income <= 50k", "income > 50k"]))
print("Decision Tree: \n", classification_report(log_pred, y_test, target_names=["income <= 50k", "income > 50k"]))
print("Random Forest: \n", classification_report(rf_pred, y_test, target_names=["income <= 50k", "income > 50k"]))


# In[ ]:


# Calculating ROC curve and AUC using sklearn for all classifer
_plot_roc(y_test, log_clf.fit(x_train, y_train).decision_function(x_test), "Logistic Regression")
_plot_roc(y_test, dt_clf.fit(x_train, y_train).predict_proba(x_test)[:,1], "Decision Tree")
_plot_roc(y_test, rf_clf.fit(x_train, y_train).predict_proba(x_test)[:,1], "Random Forest")


# This is the same code I used for the Milestone 3 Project. From the resulting ROC plots it seems that Logistic Regression has the best preformance out of the classifiers. Area under the curve is 0.9, which is pretty good. Accuracy metrics are displayed above, and f1-score is relatively high and the same for all classifers. It seems to have a easie time predicting for when income is below or equal to 50K, which makes me think that when income reaches past 50K it becomes less reliable of a predictor for marita status (really not sure about this one, please correct me if I am wrong).
