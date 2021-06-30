import sys
import os
import re
import numpy as np
import random
from TransformTweet import transform_tweets
#from transform import transform_tweets
from TrainModel import trainModel
import xlsxwriter
import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score,f1_score
from sklearn.svm import LinearSVC
import pandas as pd
import csv

#exporta los sentimientos asociados a los autores de los tuits
def exportData(archivoReferencia):
    datos = pd.read_csv(archivoReferencia, header=0)
    tuiteros = datos['Label']

    for autor in tuiteros:
        if (dic.get(autor, "null") != "null"):
            sentimientosFiltrados.append(int(dic.get(autor)))
            autoresFiltrados.append(autor)

    df = pd.DataFrame({'Sentimientos': sentimientosFiltrados, 'Autores': autoresFiltrados})
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('sentimientos.xlsx', engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Hoja 1')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    print("sentimientos exportados")

#leemos un archivo de recopilación de tuits para analizar el sentimiento de los mismos
def textSentiment(archivoLectura):
    print("Analizando tuits...")
    with open(archivoLectura, encoding="utf8", mode="r+") as fichero:
        for linea in fichero:
            linea = linea.strip('\r\n')
            data = linea.split('\t')
            tweet = data[3].lower()
            cleanText = transform_tweets(tweet)
            sent = clf.predict((tfidf.transform([cleanText])))
            data[3] = cleanText
            dic[data[2].lower()] = sent[0]
            if (data[21] != "None"):
                dic[data[21].lower()] = sent[0]
        fichero.close()
        print("Tuits analizados: "+str(len(dic)))


print('Iniciando...')
contador=0
tweets=[]
sentiments=[]
corpus='datasets/cost.csv'
#Leemos un corpus y organizamos las variables en dataFrames para usarlos como variables en nuestro modelo
with open(corpus, newline='',encoding='utf-8') as File:
    reader = csv.DictReader(File,delimiter=',',quotechar='"')
    for row in reader:
        tweets.append(row['tweet'])
        sentiments.append(row['Sentiment'])
        contador+=1

d = {'tweet': tweets, 'Sentiment': sentiments}
data_yelp=pd.DataFrame(columns=['tweet','Sentiment'],data=d)

X = data_yelp['tweet']
y = data_yelp['Sentiment']

data_yelp['tweet'] = data_yelp['tweet'].apply(lambda x: transform_tweets(x)) #Aplicamos nuestro tokenizador a los tuits
tfidf = TfidfVectorizer(tokenizer=transform_tweets,ngram_range=(1,2)) #TFIDF
X = tfidf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) #iniciamos el entrenamiento con los parámetros dado
clf = LinearSVC() #Inicializamos nuestro clasificador
clf.fit(X_train, y_train) #Ajustamos los datos al modelo
y_pred = clf.predict(X_test) #Predecimos los datos X mediante el clasificador

print(f1_score(y_test, y_pred, labels=np.unique(y_pred),average=None)) #imprimimos resultados del modelo
print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

#semillaOptima = trainModel(1000, 'datasets/cost.csv', 10000)  #método para probar la precisión de diferentes semillas

sentimientosFiltrados = []
autoresFiltrados = []
dic = {}
archivoLectura='store\\OVNI\\OVNI.txt'
textSentiment(archivoLectura)
archivoReferencia='store\\OVNI\\OVNINodos.csv'
exportData(archivoReferencia)