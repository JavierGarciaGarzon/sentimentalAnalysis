import sys
import os
import re
import emoji
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
import pandas as pd
import string

punctuations=string.punctuation
def parser(sentence):
    mytokens = parser(sentence)
    return mytokens.text

#tokenizador alternativo
def tweettok(x):
    tknzr = TweetTokenizer(reduce_len=True, strip_handles=True)
    lista=tknzr.tokenize(x)
    frase=""
    for palabra in lista:
        frase+=palabra+" "
    #print(frase)
    x=frase
    return x

# Eliminamos signos de puntuación
def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

# Corregimos abreviaciones
def fix_abbr(x):
    if type(x) == list:
        words = x
    elif type(x) == str:
        words = x.split()
    else:
        raise TypeError('El formato no es válido, debe ser lista o str')

    abbrevs = {'d': 'de',
               'x': 'por',
               'xa': 'para',
               'as': 'has',
               'q': 'que',
               'k': 'que',
               'dl': 'del',
               'xq': 'porqué',
               'dr': 'doctor',
               'dra': 'doctora',
               'sr': 'señor',
               'sra': 'señora',
               'm': 'me'}
    return " ".join([abbrevs[word] if word in abbrevs.keys() else word for word in words])

# Sustituimos links por {link}
def remove_links(text):
    text = " ".join(['' if ('http') in word else word for word in text.split()])
    return text

# Sustituimos rt por {rt}
def remove_rt(text):
     text=" ".join(['' if ('rt') in word else word for word in text.split()])
     return text

# Eliminamos vocales repetidas
def remove_repeated_vocals(text):
    list_new_word = []

    for word in text.split():  # separamos en palabras
        new_word = []
        pos = 0

        for letra in word:  # separamos cada palabra en letras
            # print(word, letra, pos, '-', new_word)
            if pos > 0:
                if letra in ('a', 'e', 'i', 'o', 'u') and letra == new_word[pos - 1]:
                    None
                else:
                    new_word.append(letra)
                    pos += 1
            else:
                new_word.append(letra)

                pos += 1
        else:
            list_new_word.append("".join(new_word))

    return " ".join(list_new_word)

# Normalizamos risas
def normalize_laughts(text,doc):
    list_new_words = []
    for word in text.split():  # separamos en palabras
        count = 0
        vocals_dicc = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0}

        for letra in word:
            # print(word)
            if letra == 'j':
                count += 1
            if letra in vocals_dicc.keys():
                vocals_dicc[letra] += 1
        else:
            if count > 3:
                dicc_risa = {'a': 'jaja', 'e': 'jeje', 'i': 'jiji', 'o': 'jojo', 'u': 'juju'}
                risa_type = max(vocals_dicc, key=lambda x: vocals_dicc[x])  # Indica si es a,e,i,o,u
                list_new_words.append(dicc_risa[risa_type])
            else:
                list_new_words.append(word)

    text= " ".join(list_new_words)
    return  text

# Eliminamos el Hastag
def remove_hashtags(text):
    text = " ".join(['{hash}' if word.startswith('#') else word for word in text.split()])
    return text

# Quitamos las menciones
def remove_mentions(text):
    text = ' '.join(token for token in text.split() if token.find("@")==-1)
    return text

# Función para identificar los 'emojis' tradicionales
def transform_icons(text):
    for token in text.split():
        if token in emoji.EMOJI_UNICODE_SPANISH.values():
            token,emoji.demojize(token,language='es')
    return text

# Devuelve true/false si es emoji
def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI


# Separamos emojis que vengan juntos
def sep_emojis(text):
    words_list = []
    for token in text.split():
        new_word = []
        for letra in token:
            if letra in emoji.UNICODE_EMOJI:
                words_list.append(letra)
            else:
                new_word.append(letra)
        else:
            words_list.append("".join(new_word))

    text=(" ".join(word for word in words_list if word != ''))
    return text
# Eliminamos stopwords
def remove_stopwords(text,doc):
    text_no_stop = [token for token in doc if not token.is_stop]
    text= ' '.join(token.text for token in text_no_stop)
    return text


###
# Función de preprocesamiento del texto
# dejo comentado aquellos métodos que no me han dado tan buenos resultados
def transform_tweets(text):
    #text=tweettok(text)
    # text = remove_stopwords(text,doc)
    text = remove_mentions(text)
    text = remove_punctuation(text)
    text = remove_links(text)
    text = remove_hashtags(text)
    text = remove_rt(text)
    #text = parser(text)
    #text = transform_icons(text)
    # text = sep_emojis(text,doc)
    # text = normalize_laughts(text,doc)
    #df = remove_repeated_vocals(df)
    #df = fix_abbr(df)
    # df = stem(df) #Opción para stemizar

    return text


