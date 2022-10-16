import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from datasets import load_dataset
import random
import os
import numpy as np 
stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()


def preprocess_and_write(df, path):
    df.to_csv(path, index=False)
    print("preprocessed dataset is ready.")

def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

def convert_to_df(data_list):
    text = []
    label = []
    for sample in data_list:
        text.append(sample['text'])
        label.append(sample['label'])
    dict_data = {'text': text, 'label': label}
    df_data = pd.DataFrame(data=dict_data)
    return df_data

df_extra = pd.read_csv('./data/imdb_master.csv', encoding="latin-1")
print(df_extra.head())
df_extra = df_extra.drop(['Unnamed: 0','type','file'],axis=1)
df_extra.columns = ["text", "label"]
df_extra = df_extra[df_extra.label != 'unsup']
df_extra['label'] = df_extra['label'].map({'pos': 1, 'neg': 0})

dataset = load_dataset("imdb")
train_data = [item for item in dataset['train']]
test_data = [item for item in dataset['test']]


df_train = convert_to_df(train_data)
df_train_dev = pd.concat([df_train, df_extra]).reset_index(drop=True)
df_test = convert_to_df(test_data)
df_train_dev['processed_text'] = df_train_dev.text.apply(lambda x: clean_text(x))
df_train_dev = df_train_dev.drop(['text'],axis=1)
df_train_dev = df_train_dev[['processed_text', 'label']]
df_train_dev.columns = ['text', 'label']

df_test['processed_text'] = df_test.text.apply(lambda x: clean_text(x))
df_test = df_test.drop(['text'],axis=1)
df_test = df_test[['processed_text', 'label']]
df_test.columns = ['text', 'label']

df_train_dev = df_train_dev.sample(frac=1).reset_index(drop=True)
df_test = df_test.sample(frac=1).reset_index(drop=True)
print(df_train_dev)
print(df_test)
df_train = df_train_dev[:70000]
df_dev = df_train_dev[70000:]
preprocessed_folder = "./preprocessed"
try:
    os.makedirs(preprocessed_folder)
except FileExistsError:
    pass

preprocess_and_write(df_train, os.path.join(preprocessed_folder,"train.csv"))
preprocess_and_write(df_dev, os.path.join(preprocessed_folder,"dev.csv"))
preprocess_and_write(df_test, os.path.join(preprocessed_folder,"test.csv"))


