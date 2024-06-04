#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

import numpy as np
import json
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import ToktokTokenizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

tokenizer = ToktokTokenizer() 
STOPWORDS = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def limpiar_texto(texto):
    """
    Función para realizar la limpieza de un texto dado.
    """
    # Eliminamos los caracteres especiales
    texto = re.sub(r'\W', ' ', str(texto))
    # Eliminado las palabras que tengo un solo caracter
    texto = re.sub(r'\s+[a-zA-Z]\s+', ' ', texto)
    # Sustituir los espacios en blanco en uno solo
    texto = re.sub(r'\s+', ' ', texto, flags=re.I)
    # Convertimos textos a minusculas
    texto = texto.lower()
    return texto

def filtrar_stopword_digitos(tokens):
    """
    Filtra stopwords y digitos de una lista de tokens.
    """
    return [token for token in tokens if token not in STOPWORDS 
            and not token.isdigit()]

def stem_palabras(tokens):
    """
    Reduce cada palabra de una lista dada a su raíz.
    """
    return [stemmer.stem(token) for token in tokens]

def tokenize(texto):
    """
    Método encargado de realizar la limpieza y preprocesamiento de un texto
    """
    text_cleaned = limpiar_texto(texto)
    tokens = [word for word in tokenizer.tokenize(text_cleaned) if len(word) > 1]
    tokens = filtrar_stopword_digitos(tokens)
    stems = stem_palabras(tokens)
    return stems

#Definición de tranformacion del texto
dataTraining = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
vect = TfidfVectorizer(tokenizer=tokenize,sublinear_tf=True,max_features=15000)
X_dtm = vect.fit_transform(dataTraining['plot']).toarray()

def predict_genero(url):

    lr_1 = joblib.load(os.path.dirname(__file__) + '/pred_genres_text_RL.pkl') 
    #vect = joblib.load(os.path.dirname(__file__) + '/pred_genres_vect.pkl') 

    url_ = pd.DataFrame([url], columns=['plot'])

    # transformación variables predictoras X del conjunto de test
    url_ = vect.transform(url_['plot'])

    colum = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

    # Predicción del conjunto de test
    pred_gene = lr_1.predict_proba(url_)

    # Guardar predicciones en formato exigido en la competencia de kaggle
    res = pd.DataFrame(pred_gene,  columns=colum)

    return res


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an TEXTO')
        
    else:

        url = sys.argv[1]

        res = predict_genero(url)
        
        print(url)
        print('Generos: ', res)
        