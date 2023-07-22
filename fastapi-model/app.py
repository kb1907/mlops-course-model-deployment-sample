from fastapi import FastAPI
import uvicorn
import pickle
import numpy as np

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Prediction"}

@app.get("/predict")
async def predict(Age:int, RestingBP:int, Cholesterol:int,FastingBS:int,MaxHR:int,Oldpeak:float):
    model = pickle.load(open('catboost_model-2.pkl', 'rb'))
    prediction = model.predict([[Age, RestingBP, Cholesterol,FastingBS,MaxHR,Oldpeak]])
    if prediction == 0:
        return {f"Prediction: {prediction}. You are well. No worries :)"}
    else:
        return {f"Prediction: {prediction}. Kindly make an appointment with the doctor!"}



if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')

