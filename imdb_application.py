import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.backend import clear_session
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

def get_vectorized_data(reviews_train, reviews_test):
    vectorizer = CountVectorizer()
    vectorizer.fit(reviews_train)
    x_train = vectorizer.transform(reviews_train)
    x_test = vectorizer.transform(reviews_test)
    return x_train, x_test


    
train_df = pd.read_csv('./preprocessed/train.csv')
dev_df = pd.read_csv('./preprocessed/dev.csv')
test_df = pd.read_csv('./preprocessed/test.csv')
all_df = pd.concat([train_df, dev_df, test_df]).reset_index(drop=True)
all_df = all_df.sample(frac=1).reset_index(drop=True)
all_df = all_df[:200]
reviews = all_df['text'].values
labels = all_df['label'].values


reviews_train, reviews_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.15, random_state=1000)
        
# logistic regression and neural networks
"""
x_train, x_test = get_vectorized_data(reviews_train, reviews_test)

classifier = LogisticRegression()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("logistic regression results: ", precision_recall_fscore_support(y_test, y_pred, average='binary'))

input_dim = x_train.shape[1]
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, epochs=3, verbose='auto',batch_size=100)
clear_session()
y_pred = model.predict(x_test, batch_size=1250)
y_pred = np.where(y_pred > 0.5, 1, 0)
print("neural networks results: ", precision_recall_fscore_support(y_test, y_pred, average='binary'))
"""
# embedding + neural networks
"""
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(reviews_train)
x_train = tokenizer.texts_to_sequences(reviews_train)
x_test = tokenizer.texts_to_sequences(reviews_test)
vocab_size = len(tokenizer.word_index) + 1 
maxlen = 100
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

embedding_dim = 128

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs=8, verbose='auto',batch_size=100, validation_split=0.0)
clear_session()
y_pred = model.predict(x_test, batch_size=1250)
y_pred = np.where(y_pred > 0.5, 1, 0)
print("embedding + neural networks results: ", precision_recall_fscore_support(y_test, y_pred, average='binary'))
"""
# cnn
"""
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(reviews_train)
x_train = tokenizer.texts_to_sequences(reviews_train)
x_test = tokenizer.texts_to_sequences(reviews_test)
vocab_size = len(tokenizer.word_index) + 1 
maxlen = 100
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

embedding_dim = 128
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.Dropout(0.05))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=6, verbose='auto',batch_size=100, validation_split=0.1)
clear_session()
y_pred = model.predict(x_test, batch_size=1250)
y_pred = np.where(y_pred > 0.5, 1, 0)
print(precision_recall_fscore_support(y_test, y_pred, average='binary'))
"""
# lstm 
"""
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(reviews_train)
x_train = tokenizer.texts_to_sequences(reviews_train)
x_test = tokenizer.texts_to_sequences(reviews_test)
vocab_size = len(tokenizer.word_index) + 1 
maxlen = 100
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

embedding_dim = 128
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           mask_zero=True))
#model.add(layers.Bidirectional(layers.LSTM(128, kernel_regularizer='l2')))
model.add(layers.LSTM(128, kernel_regularizer='l2', return_sequences=True))
model.add(layers.Dropout(0.05))
model.add(layers.LSTM(128, kernel_regularizer='l2'))
model.add(layers.Dense(10, activation='relu', kernel_regularizer='l2'))
model.add(layers.Dense(1))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=8, verbose='auto',batch_size=100, validation_split=0.1)
clear_session()
y_pred = model.predict(x_test, batch_size=1250)
y_pred = np.where(y_pred > 0.5, 1, 0)
print(precision_recall_fscore_support(y_test, y_pred, average='binary'))
"""
# pre-trained language models

reviews_train, reviews_test, y_train, y_test

all_dataset = {}
train_dict = {"text": reviews_train, "label":y_train}
test_dict = {"text": reviews_test, "label":y_test}
all_dataset['train'] = Dataset.from_dict(train_dict)
all_dataset['test'] = Dataset.from_dict(test_dict)
all_dataset = DatasetDict(all_dataset)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = all_dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
training_args = TrainingArguments(
    output_dir="./transformers-results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
)
accuracy_metric = evaluate.load('accuracy')
recall_metric = evaluate.load('recall')
precision_metric = evaluate.load('precision')
f1_metric = evaluate.load('f1')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    results = {"acc":accuracy_metric.compute(predictions=predictions, references=labels),
               "precision":precision_metric.compute(predictions=predictions, references=labels),
               "recall":recall_metric.compute(predictions=predictions, references=labels),
               "f1":f1_metric.compute(predictions=predictions, references=labels)}
    return results
    
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

