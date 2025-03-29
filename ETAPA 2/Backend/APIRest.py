from typing import Optional
from joblib import load
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import PredictionModel

app = FastAPI()


@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}


@app.get("/predecir/{path}")
def predecir(path:str):
   path = "../Data/" + path
   data=pd.read_csv(path, sep=',')
   
   model = PredictionModel.Model()
   data = model.remove_duplicates(data)
   data['Titulo'] = model.pipelinePreprocess.transform(data['Titulo'])
   data['Descripcion'] = model.pipelinePreprocess.transform(data['Descripcion'])
   X_data=  data['Titulo'].astype(str) + " " + data['Descripcion'].astype(str)
   vectores = model.Vectorizer(X_data)
   prediciones=model.make_prediction(vectores)
   return prediciones