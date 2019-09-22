#!/usr/bin/env python
# coding: utf-8

# In[22]:


from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb


# In[ ]:





# In[23]:



# Embedding
max_features = 20000
max_review_length = 500
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2


# In[24]:



'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''


# In[25]:


import numpy as np # IssueFix1 To solve ValueError: Object arrays cannot be loaded when allow_pickle=False
# IssueFix1 save np.load
np_load_old = np.load
# IssueFix1 modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


# In[26]:



print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# IssueFix1 restore np.load for future normal usage
np.load = np_load_old

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


# In[27]:



print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# In[28]:



print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=max_review_length))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[29]:



model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[30]:



print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))


# In[31]:


score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


# In[32]:


#reverse lookup
INDEX_FROM = 3
word_to_id = imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[id] for id in x_train[0] ))


# In[34]:


print(model.summary())


# In[43]:



#predict sentiment from reviews
bad = "overall its a bad movie but some scene are good"
good = "i really liked the movie and had fun"
neutral = "i did not like the movie"
for review in [good,bad, neutral]:
    tmp = []
    for word in review.split(" "):
        tmp.append(word_to_id[word])
    tmp_padded = sequence.pad_sequences([tmp], maxlen=max_review_length) 
    print("%s. Sentiment: %s" % (review,model.predict(np.array([tmp_padded][0]))[0][0]))
 


# In[ ]:




