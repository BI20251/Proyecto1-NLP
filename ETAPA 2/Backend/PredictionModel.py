import json
import unicodedata
import re
from num2words import num2words
from joblib import load
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


class RemoveNonAscii(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda text: unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore') if isinstance(text, str) else text)


class ToLowercase(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.str.lower()


class RemovePunctuation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda text: re.sub(r'[^\w\s]', '', text) if isinstance(text, str) else text)


class ReplaceNumbers(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda text: ' '.join([num2words(int(word), lang='es') if word.isdigit() else word for word in text.split()]) if isinstance(text, str) else text)


class RemoveStopwords(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stopwords = set(stopwords.words('spanish'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda text: ' '.join([word for word in text.split() if word not in self.stopwords]) if isinstance(text, str) else text)


class Model:
    def __init__(self):
        self.model = load("../assets/modelo.joblib")
        

        self.pipelinePreprocess = Pipeline([
            ('remove_non_ascii', RemoveNonAscii()),
            ('to_lowercase', ToLowercase()),
            ('remove_punctuation', RemovePunctuation()),
            ('replace_numbers', ReplaceNumbers()),
            ('remove_stopwords', RemoveStopwords())
        ])

    def remove_duplicates(self, df):
        df.drop_duplicates(subset='Titulo', keep='first', inplace=True)
        df.drop_duplicates(subset='Descripcion', keep='first', inplace=True)
        return df

    def Vectorizer(self, data):
        vectorizer = TfidfVectorizer(max_features=5000, stop_words=stopwords.words('spanish'))
        return vectorizer.fit_transform(data)

    def make_prediction(self, X_data):
        proba = self.model.predict_proba(X_data)  # Obtiene las probabilidades
        clases = self.model.classes_  # Obtiene los nombres de las clases
        
        predicciones = []
        for prob in proba:
            max_index = prob.argmax()  # Índice de la clase con mayor probabilidad
            predicciones.append({
                "clase": str(clases[max_index]),  # Convertimos a string por seguridad
                "probabilidad": float(prob[max_index])  # Convertimos a float para JSON válido
            })
        
        return json.dumps(predicciones)
