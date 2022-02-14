# Handwritten-Digit-Recognition-on-MNIST-dataset

# fetching dataset
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

mnist = fetch_openml('mnist_784')
x, y = mnist['data'], mnist['target']

some_digit = x.to_numpy()[36001]
some_digit_image = some_digit.reshape(28, 28)  # let's reshape to plot it

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
           interpolation='nearest')
plt.axis("off")
plt.show()

x_train, x_test = x[:60000], x[6000:70000]
y_train, y_test = y[:60000], y[6000:70000]

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train.[shuffle_index], y_train.[shuffle_index]

# Creating a 2-detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == '2')
y_test_2 = (y_test == '2')

# Train a logistic regression classifier
clf = LogisticRegression(tol=0.1)
clf.fit(x_train, y_train_2)
example = clf.predict([some_digit])
print(example)

# Cross Validation
a = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")
print(a.mean())


KNN 


#!/usr/bin/env python
# coding: utf-8

# # Handwritten Digit Recognition on MNIST dataset

# MNIST = “Modified National Institute of Standards and Technology”
>>sets of 70,000 small images of data and handwritten by high school students and employees of the US census bureau
>>All images are labelled with the respective digits they represent
>>MNIST is  the "Hello World" of machine learning
>>There are 70,000 images and its image has 784 features
>>Each image is 28 *28 pixels and each feature simply represents on pixel's intensity from 0 (white) to 255 (black)
>> Image is collection of pixels
# >>K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique.
# >>K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.
# >>K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.
# >>K-NN algorithm can be used for Regression as well as for Classification but mostly it is used for the Classification problems.

# # Importing libraries

# In[1]:


import numpy as np #array 
import pandas as pd #calculation
import matplotlib.pyplot as plt #vis
#Seaborn is a Python data visualization library based on matplotlib. 
#It provides a high-level interface for drawing attractive and informative statistical graphics.
import seaborn as sns
#only draw static images in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the MNIST datasets

# In[27]:


train_df = pd.read_csv(r"C:\Users\LEGION\Desktop\edubridge\DA\p1\digit-recognizer\train.csv")
test_df = pd.read_csv(r"C:\Users\LEGION\Desktop\edubridge\DA\p1\digit-recognizer\test.csv")


# In[3]:


train_df.head()


# In[4]:


train_df.tail()


# In[42]:


train_df.info


# In[34]:


train_df.info()


# In[28]:


train_df.describe


# In[35]:


train_df.describe()


# In[29]:


test_df.info


# In[44]:


test_df.head()


# # For train and test both we will use train.csv (Taking train data as complete data)

# In[7]:


train_df.shape


# # Data Preparation for Model Building

# In[8]:


y=train_df['label']
x=train_df.drop('label',axis=1)


# In[32]:


x_for_test_data=test_df[:]


# In[51]:


#The noisy MNIST datasets. Examples of noisy MNIST samples:
#background image noise (MNISTbi), random noise (MNISTbi), the variational noises.
plt.figure(figsize=(3,3))
some_digit=151
some_digit_image = x.iloc[some_digit].to_numpy().reshape(28, 28)
plt.imshow(np.reshape(some_digit_image, (28,28)), cmap=plt.cm.gray)
plt.axis('off')
print(y[some_digit])


# In[13]:


plt.figure(figsize=(9,9))
some_digit=160
some_digit_image = x.iloc[some_digit].to_numpy().reshape(28, 28)
plt.imshow(np.reshape(some_digit_image, (28,28)), cmap=plt.cm.gray)
print(y[some_digit])


# In[14]:


plt.figure(figsize=(9,9))
some_digit=10
some_digit_image = x.iloc[some_digit].to_numpy().reshape(28, 28)
plt.imshow(np.reshape(some_digit_image, (28,28)), cmap=plt.cm.gray)
print(y[some_digit])


# In[15]:


#Show the counts of observations in each categorical bin using bars.
#A count plot can be thought of as a histogram across a categorical, instead of quantitative, variable. 
#The basic API and options are identical to those for barplot(), so you can compare counts across nested variables.
sns.countplot(train_df['label'])


# # Splitting the train data into train and test

# In[16]:


from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)


# # Models

# # KNN

# In[17]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# k=5

# In[47]:


#Scikit-learn is an open source machine learning library that supports supervised and unsupervised learning.
#It also provides various tools for model fitting, 
#data preprocessing, model selection, model evaluation, and many other utilities.

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 21)
classifier.fit(x_train, y_train)


# In[48]:


y_pred = classifier.predict(x_test)
y_pred


# In[49]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:





