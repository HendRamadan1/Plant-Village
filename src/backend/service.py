
import torch 
import os 
import torch.nn as nn
import json
from src.utlis.preprocessing import preprocessing_image
from src.Config import Config
from torchvision import models
from collections import OrderedDict
class PredictionService:
    
    def __init__(self):
        self.device=torch.device('cuda'if torch.cuda.is_available() else "cpu")
        self.model=None
        self.classes=None


    def load_model(self):
        if not os.path.exists(Config.LABELS_PATH):
            raise FileNotFoundError()
        with open (Config.LABELS_PATH,'r') as f:
            self.classes=json.load(f)


        self.model=models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, len(self.classes)) 
        )
        state_dict = torch.load(Config.MODEL_WEIGHTS_PATH, map_location=self.device,weights_only=True)

        raw_state_dict=state_dict['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in raw_state_dict.items():
            name = k.replace("resnet.", "") 
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)


        self.model.to(self.device)
        self.model.eval()
        print("suceess load model ")



    def predict(self,image_byte):
        if self.model is None :
            print("🔄 Model not loaded yet, loading now...")
            self.load_model()
        tensor = preprocessing_image(image_bytes=image_byte).to(self.device)
        with torch.no_grad():
            outputs=self.model(tensor)
            probabilities=torch.nn.functional.softmax(outputs,dim=1)
            confidence_level,prediction=torch.max(probabilities,1)

        return {
            "class_name":self.classes[prediction.item()],
            "confidence" :f"{confidence_level.item()*100:.2f}%"
        }

predictor=PredictionService()

























