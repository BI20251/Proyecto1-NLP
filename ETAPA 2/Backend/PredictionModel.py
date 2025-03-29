import json
import unicodedata
import re
from fastapi import HTTPException
from num2words import num2words
from joblib import dump, load
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
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
        proba = self.model.predict_proba(X_data) 
        clases = self.model.classes_ 
        
        predicciones = []
        for prob in proba:
            max_index = prob.argmax()  
            predicciones.append({"c": str(clases[max_index]), "p": float(prob[max_index])})
        
        return json.dumps(predicciones)
    
    def reentrenar_modelo(self, data):
        data = self.remove_duplicates(data)
        y_true = data['Label']
        data['Titulo'] = self.pipelinePreprocess.transform(data['Titulo'])
        data['Descripcion'] = self.pipelinePreprocess.transform(data['Descripcion'])
        X_data = data['Titulo'].astype(str) + " " + data['Descripcion'].astype(str)
        X1_train, X1_test, y1_train, y1_test = train_test_split(X_data, y_true, test_size=0.2, random_state=42)
        
        X1_train_tfidf = self.Vectorizer(X1_train)
        X1_test_tfidf = self.Vectorizer(X1_test)
        self.model.fit(X1_train_tfidf, y1_train)

        y1_pred = self.model.predict(X1_test_tfidf)
        accuracy = accuracy_score(y1_test, y1_pred)
        precision = precision_score(y1_test, y1_pred, pos_label=1)
        recall = recall_score(y1_test, y1_pred, pos_label=1)
        f1 = f1_score(y1_test, y1_pred, pos_label=1)
        report = classification_report(y1_test, y1_pred, output_dict=True)

        return {"Acuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1, "Report": report}
