import pandas as pd
import re
import sqlite3

# # import csv files
# kamus abusive
abusive_data = pd.read_csv("csv_data/abusive.csv")
# kamus alay
kamus_alay_data = pd.read_csv("csv_data/alay.csv", encoding="latin-1")
kamus_alay_data.columns = ['alay', 'baku']
kamus_alay_dict = dict(zip(kamus_alay_data['alay'], kamus_alay_data['baku']))

# function to censor abusive words
def abusive_censor(text):
    for word in abusive_data['ABUSIVE']:
        if word == text:
            text = text.replace(word, '*' * 3)
    return text
def abusive_cleansing(text):
    clean_text = ' '.join([abusive_censor(i) for i in text.split()])
    return clean_text

# function to replace alay words
def alay_replace(text):
    for key, value in kamus_alay_dict.items():
        if key == text:
            text = text.replace(key, value)
    return text
def alay_cleansing(text):
    clean_text = ' '.join([alay_replace(j) for j in text.split()])
    return clean_text

# function to clean sentences
def cleansing_text(text):
    # lowercase
    clean_text = str(text).lower()
    # clean URL
    clean_text = re.sub(r'(http\S+|www\S+)', '', clean_text)
    # clean punctuations
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_text)
    # clean multiple whitespace
    clean_text = ' '.join(clean_text.split())
    return clean_text

df = pd.read_csv(
    "train_preprocess.txt",
    sep="\t",
    names=["kalimat", "sentiment"]
)

df['clean text'] = df['kalimat'].apply(cleansing_text)
df['clean abusive'] = df['clean text'].apply(abusive_cleansing)
df['clean alay'] = df['clean text'].apply(alay_cleansing)
df['clean abusive alay']=df['clean alay'].apply(abusive_cleansing)

def create_connection():
    conn = sqlite3.connect('clean_data.db')
    return conn

df.to_csv("clean_data", sep='\t')