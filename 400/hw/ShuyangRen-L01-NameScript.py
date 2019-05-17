#!/usr/bin/env python
# coding: utf-8

# This is my submission for the first assignment of the course. Here is a comment for the whole script.

# In[23]:


def my_name(): #returns my name, which is Shuyang Ren (first, last)
    return("Shuyang Ren")


# In[24]:


print(my_name())


# In[25]:


import datetime as dt


# In[26]:


def date_and_time(): #returns the current date and time
    return("The current date and time is: " + str(dt.datetime.now())) 
#since assignment did to specify how percise the time should be, I made it so it returns info down to the milisecond


# In[27]:


print(date_and_time())

