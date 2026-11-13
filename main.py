from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal # custom datatype
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# from fastpi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"

INTERFACE_DIR = BASE_DIR / "interface"
ASSETS_DIR = BASE_DIR / "assets"

model = joblib.load(
    MODEL_PATH)
top_countries = [
    'Other',
    'India',
    'USA',
    'Canada',
    'Australia',
    'UK',
    'Germany',
    'Mexico',
    'Turkey',
    'France',
]

app = FastAPI()  # create API object

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# images and other assets
app.mount(
    "/assets",
    StaticFiles(directory=ASSETS_DIR),
    name="assets"
)

# serves the frontend
app.mount(
    "/",
    StaticFiles(directory=INTERFACE_DIR, html=True),
    name="frontend"
)

# create a first Pydantic Model - data validation is being performed
class StudentData(BaseModel): # defines structured data models
    age: int = Field(..., ge=10, le=88) # boundaries: age must be >= (greater than or) 10 and <= 88 (less than or)
    gender: Literal['Female', 'Male']
    country: str
    academic_level: Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform: Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter', 
    'YouTube','TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat']
    purpose_of_use: Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours: float = Field(..., ge=0, le=24)
    daily_unlocks: int = Field(..., ge=0)
    study_hours: float = Field(..., ge=0, le=24)
    physical_activity_hours: float = Field(..., ge=0, le=24)
    sleep_hours_per_night: float = Field(..., ge=0, le=24)
    stress_level: Literal['Low', 'Medium', 'High', 'Very High']

# describe what we send back
class PredictionResponse(BaseModel):
    predicted_mental_health_score:float
    # 6.77 -> float

# @app.get('/')  # homepage (default)
# def greet():
#     return {'Mental Health Score Predictor'}

# to call in terminal: uvicorn main (name of the main file):app (name of the app object) --reload

@app.post('/predict', response_model=PredictionResponse)
# create a POST route
def predict(data: StudentData):  # upcoming data input by user
    country_group = data.country if data.country in top_countries else 'Other'
    input_row = pd.DataFrame([{
        'Age': data.age,
        'Gender': data.gender,
        'Country': data.country,
        'Academic_Level': data.academic_level,
        'Most_Used_Platform': data.most_used_platform,
        'Purpose_Of_Use': data.purpose_of_use,
        'Avg_Daily_Usage_Hours': data.avg_daily_usage_hours,
        'Daily_Unlocks': data.daily_unlocks,
        'Study_Hours': data.study_hours,
        'Physical_Activity_Hours': data.physical_activity_hours,
        'Sleep_Hours_Per_Night': data.sleep_hours_per_night,
        'Stress_Level': data.stress_level,
        'Grouped_Country': country_group,
    }])
    
    predction = model.predict(input_row)[0] # 6.77
    return PredictionResponse(predicted_mental_health_score=round(float(predction), 2))