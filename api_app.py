import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import json
model = pickle.load(open("randomforestmodel.pkl", "rb"))
app = FastAPI()
class model_input(BaseModel):
    moving_average: float
    rolling_median: float
@app.post('/volume_prediction')
def v_pred(input: model_input):
    input_data = input.json()
    input_dict = json.loads(input_data)
    mov_avg = input_dict['moving_average']
    rol_med = input_dict['rolling_median']
    input_list = [mov_avg, rol_med]
    prediction = model.predict([input_list])
    return prediction[0]

