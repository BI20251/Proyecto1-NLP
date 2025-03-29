import shutil
from fastapi import UploadFile, File
from typing import Optional
from joblib import load
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import PredictionModel
import json
import shutil
import os
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}


@app.post("/upload")
async def upload_file(file: UploadFile = File()):
    try:
        ruta = f"../Data/{file.filename}"
        with open(ruta, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"mensaje": "Archivo guardado", "filename": file.filename}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/predecir/{path}")
def predecir(path:str):
   path = "../Data/" + path
   data=pd.read_csv(path, sep=';')
   
   model = PredictionModel.Model()
   data = model.remove_duplicates(data)
   data['Titulo'] = model.pipelinePreprocess.transform(data['Titulo'])
   data['Descripcion'] = model.pipelinePreprocess.transform(data['Descripcion'])
   X_data=  data['Titulo'].astype(str) + " " + data['Descripcion'].astype(str)
   vectores = model.Vectorizer(X_data)
   predicciones=model.make_prediction(vectores)
   return {"predicciones": predicciones}



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        path = f"../Data/{file.filename}"
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"mensaje": "Archivo guardado con éxito", "filename": file.filename}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)