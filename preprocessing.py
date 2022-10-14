import pandas as pd
import os
import spacy
import random
import re
import numpy as np
import string
import nltk
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import names 
import sys
from unicodedata import category
from cleantext import clean
from datasets import load_dataset

punctuations =  [chr(i) for i in range(sys.maxunicode) if category(chr(i)).startswith("P")]
nltk.download('names')
male_names = names.words('male.txt')
female_names = names.words('female.txt')

all_names = []
for name in male_names:
    all_names.append(name.lower())
for name in female_names:
    all_names.append(name.lower())

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
nltk.download('wordnet')
nltk.download('omw-1.4')
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('punkt')


def tokenizer(text):
    return word_tokenize(text)[:500]

def remove_punctuation(tokens):
    no_punctiation = [token for token in tokens if token not in punctuations]
    return no_punctiation

def remove_stopwords(tokens):
    no_stopwords= [token for token in tokens if token not in stopwords]
    return no_stopwords

def standardize_names(tokens):
    standardized = [token if token not in all_names else '< PERSON >' for token in tokens]
    return standardized

def lemmatizer(tokens): # i prefer lemmatization over stemming since it preserves the meaning more
    lemmatized = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized

def tokens_to_text(tokens):
    text = "".join([token+" " for token in tokens])[:-1]
    return text

def general_cleaning(text):
    return clean(text,
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lower=True,                     # lowercase text
    no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=True,                # replace all email addresses with a special token
    no_phone_numbers=True,         # replace all phone numbers with a special token
    no_numbers=True,               # replace all numbers with a special token
    no_digits=True,                # replace all digits with a special token
    no_currency_symbols=True,      # replace all currency symbols with a special token
    no_punct=True,                 # remove punctuations
    replace_with_punct="",          # instead of removing punctuations you may replace them
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"                       # set to 'de' for German special handling
)
def preprocess_and_write(df, path):

    df['text'] = df['text'].apply(lambda x:general_cleaning(x))
    print("sequence has been cleaned with clean-text library.")
    df['text'] = df['text'].apply(lambda x:tokenizer(x))
    print("sequence has been tokenized.")
    df['text']= df['text'].apply(lambda x:remove_stopwords(x))
    print("stop words has been removed.")
    df['text']= df['text'].apply(lambda x:standardize_names(x))
    print("names has been standardized.")
    df['text']= df['text'].apply(lambda x:lemmatizer(x))
    print("sequence has been lemmatized.")
    df['text']= df['text'].apply(lambda x:tokens_to_text(x))
    print("sequence has been converted to text again.")
    df.to_csv(path, index=False)
    print("preprocessed dataset is ready.")

def convert_to_df(data_list):
    text = []
    label = []
    for sample in data_list:
        text.append(sample['text'])
        label.append(sample['label'])
    dict_data = {'text': text, 'label': label}
    df_data = pd.DataFrame(data=dict_data)
    return df_data

dataset = load_dataset("imdb")
train_dev_data= [item for item in dataset['train']]
random.shuffle(train_dev_data)
test_data = [item for item in dataset['test']]
random.shuffle(test_data)
train_data = train_dev_data[:10000]
dev_data = train_dev_data[10000:12000]

df_train = convert_to_df(train_data)
df_dev = convert_to_df(dev_data)
df_test = convert_to_df(test_data)

preprocessed_folder = "./preprocessed"
try:
    os.makedirs(preprocessed_folder)
except FileExistsError:
    pass

preprocess_and_write(df_train, os.path.join(preprocessed_folder,"train.csv"))
preprocess_and_write(df_dev, os.path.join(preprocessed_folder,"dev.csv"))
preprocess_and_write(df_test, os.path.join(preprocessed_folder,"test.csv"))


