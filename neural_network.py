"""
Function untuk membersihkan data text
"""

import re
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Open Neural Network Model
count_vect = pickle.load(open("feature.p", 'rb'))
with open("model.p", "rb") as f:
    lr = pickle.load(f)

def text_cleansing(text):
    # lowercase
    clean_text = str(text).lower()
    # Clean URL
    clean_text = re.sub(r'(http\S+|www\S+)', '', clean_text).strip()
    # Clean Emoticon Byte
    clean_text = clean_text.replace("\\", " ")
    clean_text = re.sub('x..', ' ', clean_text)
    clean_text = re.sub(' n ', ' ', clean_text)
    clean_text = re.sub('\\+', ' ', clean_text)
    clean_text = re.sub('  +', ' ', clean_text)
    # Clean Punctuation (Except words and number)
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_text)
    # Clean Username
    clean_text = re.sub('user',' ',clean_text)
    # Clean Multiple Whitespace
    clean_text = ' '.join(clean_text.split())
    return clean_text

# Neural Network Sentiment Analysis
def neural_sentiment(text):
    # Clean Text
    clean_text = text_cleansing(text)

    # Sentiment analysis
    text_vect = count_vect.transform([clean_text])
    sentiment_result = lr.predict(text_vect)[0]
    return sentiment_result

def neural_files(file_upload):
    # Get only the first column
    df_upload = pd.DataFrame(file_upload.iloc[:,0])
    
    # Rename column to "raw_text"
    df_upload.columns = ["raw_text"]

    # Clean text with text_cleansing function
    # Save to "clean_text" column
    df_upload["clean_text"] = df_upload["raw_text"].apply(text_cleansing)
    
    # Neural network sentiment analysis
    df_upload["sentiment"] = df_upload["clean_text"].apply(neural_sentiment)
    print("Neural network sentiment analysis success!")
    return df_upload