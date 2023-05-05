from flask import Flask, redirect, url_for, render_template, request, Response
from flask_restful import Resource, Api
import pickle
import os
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)

# Defining final data path
parent_dir = os.getcwd()
@app.route("/")
def welcome():
    Moving_average = request.args.get('Moving_average')
    Rolling_median = request.args.get('Rolling_median')
    return render_template("index.html", Moving_average = Moving_average,Rolling_median = Rolling_median)

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        Moving_average = request.form["MA"]
        Rolling_median = request.form["AJ"]
        model = pickle.load(open(parent_dir+ "/randomforestmodel.pkl", "rb"))
        prediction = model.predict([[Moving_average,Rolling_median]])
        prediction = int(prediction[0])
        return render_template("index.html", Moving_average = Moving_average,Rolling_median = Rolling_median,prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True)
