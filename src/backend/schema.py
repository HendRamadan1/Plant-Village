from pydantic import BaseModel
from sqlmodel import Field

class PredictionResult(BaseModel):
    class_name:str
    confidence : str


