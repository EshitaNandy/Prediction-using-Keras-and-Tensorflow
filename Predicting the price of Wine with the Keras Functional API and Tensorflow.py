#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement:
# 
# # "Predicting the price of Wine with the Keras Functional API and Tensorflow"
# 
# - Building a wide and deep network using Keras (tf.Keras) to predict the price of wine from its description.
# - Dataset available at https://github.com/EshitaNandy/Prediction-using-Keras-and-Tensorflow/

# - The overall goal is to create a model that can identify the variety, winery and location of a wine based on a description.
# 
# - Prerequisites : Jupyter Notebook, Pandas, Numpy, Scikitlearn and Keras (Tensorflow)

# ## Here are all the imports that we will require to build this model.

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:


import itertools
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
layers = keras.layers


# In[3]:


## This code is for testing purpose of proper Tensorflow Execution
print("You have an IT job", tf.__version__)


# In[4]:


data = pd.read_csv(r"C:\Users\User\Downloads\KERAS PYTHON\winemag-data-130k-v2.csv")
data.head()


# ## Do some preprocessing to remove null values and drop unnecessary columns:

# In[5]:


data = data[pd.notnull(data['country'])]
data = data[pd.notnull(data['price'])]
data = data.drop(data.columns[0], axis =1)
data.head()


# ## Do some preprocessing to limit wine varieties in the dataset

# In[6]:


variety_threshold = 500 ## Anything that occurs less than this will be removed
value_counts = data['variety'].value_counts()
to_remove = value_counts[value_counts <= variety_threshold].index
data.replace(to_remove,np.nan, inplace=True)
data = data[pd.notnull(data['variety'])]
data.head()


# ## Split the Data into Training Dataset and Testing Dataset

# In[7]:


train_size = int(len(data)* 0.8)
print("Train size: %d" % train_size) 
print("Train size: %d" % (len(data) - train_size))


# ## Training and Testing on Features and Labels

# In[8]:


#Train Features
description_train = data['description'][:train_size]
variety_train = data['variety'][:train_size]

#Train Labels
labels_train = data['price'][:train_size]

#Test Features
description_test = data['description'][train_size:]
variety_test = data['variety'][train_size:]

#Test Labels
labels_test = data['price'][train_size:]


# ## Create a tokenizer to preprocess our text description.

# In[9]:


vocab_size = 12000 # this is a hyperparameter
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(description_train) # only fit on train


# ## Wide Feature 1: sparse bag of words(bow) vocab_size vector

# In[10]:


description_bow_train = tokenize.texts_to_matrix(description_train)
description_bow_test = tokenize.texts_to_matrix(description_test)


# ## Wide Feature 2: one-hot vector of variety categories
# 
# - Use sklearn utility to convert label strings to numbered index

# In[11]:


import sklearn
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(variety_train)
variety_train = encoder.transform(variety_train)
variety_test = encoder.transform(variety_test)
num_classes = np.max(variety_train)+1


# - Convert labels to one hot

# In[12]:


variety_train = keras.utils.to_categorical(variety_train, num_classes)
variety_test = keras.utils.to_categorical(variety_test, num_classes)


# ## Defining our wide model with the functional API

# In[13]:


bow_inputs = layers.Input(shape=(vocab_size,))
variety_inputs = layers.Input(shape=(num_classes,))
merged_layer = layers.concatenate([bow_inputs,variety_inputs])
merged_layer = layers.Dense(256, activation='relu')(merged_layer)
predictions = layers.Dense(1)(merged_layer)
wide_model = keras.Model(inputs=[bow_inputs, variety_inputs], outputs=predictions)


# In[14]:


wide_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print(wide_model.summary())


# ## Deep Model Feature : Word Embeddings of Wine Description.

# In[15]:


train_embed = tokenize.texts_to_sequences(description_train)
test_embed = tokenize.texts_to_sequences(description_test)
max_seq_length = 170
train_embed = keras.preprocessing.sequence.pad_sequences(
    train_embed, maxlen=max_seq_length, padding="post")
test_embed = keras.preprocessing.sequence.pad_sequences(
    test_embed, maxlen=max_seq_length, padding="post")


# ## Define our Deep Model with the Functional API

# In[16]:


deep_inputs = layers.Input(shape=(max_seq_length,))
embedding = layers.Embedding(vocab_size, 8, input_length=max_seq_length)(deep_inputs)
embedding = layers.Flatten()(embedding)
embed_out = layers.Dense(1)(embedding)
deep_model = keras.Model(inputs=deep_inputs, outputs = embed_out)
print(deep_model.summary())


# In[17]:


deep_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# ## Combine wide and deep into one model

# In[18]:


merged_out = layers.concatenate([wide_model.output, deep_model.output])
merged_out = layers.Dense(1)(merged_out)
combined_model = keras.Model(wide_model.input + [deep_model.input], merged_out)
combined_model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
print(combined_model.summary())


# ## Run Training

# In[19]:


# Training
combined_model.fit([description_bow_train, variety_train] + [train_embed], labels_train, epochs=10, batch_size=128)

# Evaluation
combined_model.evaluate([description_bow_test, variety_test] + [test_embed], labels_test, batch_size=128)


# ## Generating predictions on our trained model

# In[20]:


predictions = combined_model.predict([description_bow_test, variety_test] + [test_embed])


# ## Compare predictions with actual value for the first few items in our test dataset

# In[21]:


num_predictions = 40
diff = 0

for i in range(num_predictions):
    val = predictions[i]
    print(description_test.iloc[i])
    print('Predicted Price: ',val[0], 'Actual Price: ', labels_test.iloc[i], '\n')
    diff += abs(val[0] - labels_test.iloc[i])


# ## Compare the average difference between Actual price and the Model's Predicted price

# In[23]:


print('Average Prediction Difference: ', diff/num_predictions)


# ## So, the average prediction difference is around 13Dollars in every bottle of wine which is really a good value to move ahead with. 
