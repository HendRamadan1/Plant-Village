from pydantic_settings import BaseSettings,SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    
    PROJECT_NAME:str ="plant Village Disease Classifier"
    MODEL_WEIGHTS_PATH :str ="src/save_model/resnet50_plant_model (1).pth"
    LABELS_PATH:str="src/save_model/labels.json"
    BATCH_SIZE:int = 32
    IMAGE_SIZE:int = 244
    SEED :int= 42
    NORM_MEAN :List= [0.485, 0.456, 0.406]
    NORM_STD :List= [0.229, 0.224, 0.225]
    model_config=SettingsConfigDict(
        env_file='.env',
        extra='ignore'
    )
Config=Settings()
