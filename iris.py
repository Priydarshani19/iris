#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn.datasets import load_iris
iris = load_iris()


# In[2]:


dir(iris)


# In[3]:


iris.feature_names


# In[5]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[6]:


#add target var columns
df['target']=iris.target
df.head()


# In[7]:


#name of target variables
iris.target_names


# In[9]:


df[df.target==2].head()


# In[10]:


df["flower_name"]=df.target.apply(lambda x: iris.target_names[x])
df


# In[11]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]


# In[13]:


df2.head()


# In[14]:


#lets drow scatterplot for sepal length and width
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.scatter(df0["sepal length (cm)"],df0["sepal width (cm)"],color="green",marker="+")
plt.scatter(df1["sepal length (cm)"],df1["sepal width (cm)"],color="blue",marker=".")


# In[15]:


#lets drow scatterplot for petal length and width
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.scatter(df0["petal length (cm)"],df0["petal width (cm)"],color="green",marker="+")
plt.scatter(df1["petal length (cm)"],df1["petal width (cm)"],color="blue",marker=".")


# In[16]:


#lets train model using sklearn
from sklearn.model_selection import train_test_split


# In[17]:


x=df.drop(['target',"flower_name"],axis="columns")
x


# In[20]:


y=df['target']
y


# In[23]:


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)


# In[24]:


len(x_train)


# In[25]:


len(x_test)


# In[30]:


from sklearn.svm import SVC
model=SVC()


# In[31]:


model.fit(x_train,y_train)


# In[32]:


model.score(x_test,y_test)


# In[33]:


model.predict([[5.1,3.5,1.4,0.2]])


# In[ ]:




