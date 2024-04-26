import torch
import numpy as np
import torch.nn as nn
from joblib import load
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = load('scaler.joblib')

class HousePriceModel(nn.Module):
    def __init__(self, input_size):
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model = HousePriceModel(5)
model.load_state_dict(torch.load("models/house_price_model.pth", map_location=device))
model.to(device)
model.eval()

class HouseFeatures(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    furnishing: int
    parking: int

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def getStatus():
    return {"status": "OK"}

@app.post("/predict_price/")
async def predict(features: HouseFeatures):
    try:
        input_data = np.array([[features.area, features.bedrooms, features.bathrooms, features.furnishing, features.parking]])
        scaled_data = scaler.transform(input_data)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor).to(device)
        return {"predicted_price": output.item()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
