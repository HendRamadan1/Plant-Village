import pandas as pd
import os 
import numpy as np 
import torch 
import seaborn as sns
from collections import Counter 
from torchvision import datasets ,transforms
from torch.utils.data import DataLoader, Subset
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import torch.optim as optim 
from PIL import Image
import io
from torchvision  import models
from src.Config import Config

    
def preprocessing_image(image_bytes:bytes):
    image =Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform=transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE,Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(Config.NORM_MEAN,Config.NORM_STD)
    ])
    return transform(image).unsqueeze(0)

