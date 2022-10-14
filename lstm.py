import random
import pandas as pd
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten, Bidirectional, GlobalMaxPool1D, Convolution1D
from keras.models import Model, Sequential
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def df_to_tokenized_array(df, tokenizer, max_len):
    tokenizer.fit_on_texts(df['text'].values)
    tokenized_text = tokenizer.texts_to_sequences(df['text'].values)
    tokenized_text_array = pad_sequences(tokenized_text, maxlen=max_len, padding='post', truncating='post')
    label_array = df['label'].to_numpy()
    return tokenized_text_array, label_array

def lstm(max_features, embed_size):
    model = Sequential()
    model.add(Embedding(max_features, embed_size))
    model.add(Bidirectional(LSTM(32, return_sequences = True)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(1, activation="sigmoid"))
    return model
    
max_len = 200
embed_size = 128
batch_size = 1250
learning_rate = 0.001
max_features = 6000
tokenizer = Tokenizer(num_words=max_features, split=' ') 

train_df = pd.read_csv('./preprocessed/train.csv')
dev_df = pd.read_csv('./preprocessed/dev.csv')
test_df = pd.read_csv('./preprocessed/test.csv')

x_train, y_train = df_to_tokenized_array(train_df, tokenizer, max_len)
x_dev, y_dev = df_to_tokenized_array(dev_df, tokenizer, max_len)
x_test, y_test = df_to_tokenized_array(test_df, tokenizer, max_len)

model = lstm(max_features, embed_size)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, validation_data=(x_dev, y_dev), epochs = 10, batch_size=batch_size, verbose = 'auto')

y_pred = model.predict(x_test, batch_size=batch_size)
y_pred = np.argmax(y_pred, axis=1)
#y_test = np.argmax(y_test, axis=1)

print(precision_recall_fscore_support(y_test, y_pred, average='binary'))