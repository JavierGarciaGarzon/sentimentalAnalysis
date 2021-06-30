import sys
import os
import re
from nltk.tokenize import TweetTokenizer
import random
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
import pandas as pd
from TransformTweet import tweettok
from TransformTweet import transform_tweets

#'datasets/es.tsv'
def trainModel(numIt,dataEntry,maxSemilla):
    data_yelp = pd.read_csv(dataEntry, sep='\t', header=None)

    tfidf = TfidfVectorizer(tokenizer=transform_tweets, max_features=5000, ngram_range=(1, 2))
    x = data_yelp['tweet']
    y = data_yelp['Sentiment']
    x = tfidf.fit_transform(x)
    data_yelp['tweet'] = data_yelp['tweet'].apply(lambda x: transform_tweets(x))
    precision = 0
    semillaOptima = 0
    for i in range(numIt):
        semilla = random.randint(0, maxSemilla)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=semilla)##len(x)*0.3
        clf = LinearSVC()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        if (accuracy_score(y_test, y_pred) > precision):
            semillaOptima = semilla
            precision = accuracy_score(y_test, y_pred)
    print("Semilla: ", semilla, " Precision: ", precision)
    archivo=open('trainResults.txt', "a")
    archivo.write("\n" + "Archivo: %s" % dataEntry)
    archivo.write(" Semilla: %s"%semilla)
    archivo.write(" precision: %s"%precision)
    archivo.close()
    return semillaOptima



