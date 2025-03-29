from pydantic import BaseModel

class DataModel(BaseModel):
    ID: float
    Titulo:str
    Descripcion: str
    Fecha: str


    def columns(self):
        return ['ID', 'Titulo', 'Descripcion', 'Fecha']
