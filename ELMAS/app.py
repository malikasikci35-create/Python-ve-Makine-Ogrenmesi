# Web sunucusu (Uvicorn) ve FastAPI kütüphanelerini yüklüyoruz
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd

# FastAPI uygulamasını başlatıyoruz
app = FastAPI()

# HTML şablonlarının (UI) bulunacağı klasörü tanımlıyoruz
templates = Jinja2Templates(directory="templates")

# Önceden kaydettiğimiz model, encoder ve scaler nesnelerini diske erişerek yüklüyoruz
with open('30-diamond_model_complete.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    encoders = saved_data['encoders']
    scaler = saved_data['scaler']

# Kullanıcıdan/Kullanıcı arayüzünden gelecek verinin tipini ve formatını doğrulayan Pydantic veri şeması
class DiamondFeatures(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

    # Ana sayfa rotası: Kullanıcı siteye girdiğinde index.html şablonunu döndürür
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    # Tahmin rotası: Kullanıcıdan gelen verilerle elmas fiyatı tahmini yapar
    @app.post("/predict")
    async def predict(features: DiamondFeatures):
        # 1. Kullanıcıdan gelen JSON formatındaki veriyi pandas DataFrame'e çeviriyoruz
        input_data = pd.DataFrame([features.model_dump()])

        # 2. Kaydedilen encoder'lar ile kategorik sütunları sayısal değerlere dönüştürüyoruz (Label Encoding)
        for col in ['cut', 'color', 'clarity']:
            input_data[col] = encoders[col].transform(input_data[col])

        # 3. Sayısallaştırılan veriyi kaydedilen scaler ile ölçeklendiriyoruz (Standard Scaling)
        input_scaled = scaler.transform(input_data)

        # 4. İşlenmiş veri ile makine öğrenmesi modelinden tahmini alıyoruz
        prediction = model.predict(input_scaled)[0]

        # 5. Tahmin edilen elmas fiyatını JSON formatında istemciye yanıt olarak dönüyoruz
        return {"predicted_price": float(prediction)}