import random
import pandas as pd
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

train_df = pd.read_csv('./preprocessed/train.csv')
dev_df = pd.read_csv('./preprocessed/dev.csv')
test_df = pd.read_csv('./preprocessed/test.csv')

tokenizer = Tokenizer(num_words=500, split=' ') 
def df_to_tokenized_array(df):
    tokenizer.fit_on_texts(df['text'].values)
    tokenized_text = tokenizer.texts_to_sequences(df['text'].values)
    tokenized_text_array = pad_sequences(tokenized_text, maxlen=500, padding='post', truncating='post')
    label_array = pd.get_dummies(df['label']).to_numpy()
    return tokenized_text_array, label_array
  

x_train, y_train = df_to_tokenized_array(train_df)
x_dev, y_dev = df_to_tokenized_array(dev_df)
x_test, y_test = df_to_tokenized_array(test_df)


model = Sequential()
model.add(Embedding(500, 120, mask_zero=True, input_length = x_train.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(176, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
adam = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss = 'binary_crossentropy', optimizer=adam, metrics = ['accuracy'])
print(model.summary())
batch_size = 1250
model.fit(x_train, y_train, validation_data=(x_dev, y_dev), epochs = 10, batch_size=batch_size, verbose = 'auto')

y_pred = model.predict(x_test, batch_size=batch_size)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)


print(precision_recall_fscore_support(y_test, y_pred, average='binary'))