from flask import Flask, request, render_template

import sys
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler

from src.utils import load_object

from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

from src.exception import CustomException

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods =['GET','POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            data=CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=int(request.form.get('reading_score')),
                writing_score=int(request.form.get('writing_score'))
            )

            pred_df=data.get_data_as_frame()

            pred_pipeline = PredictionPipeline()

            results = pred_pipeline.predict(pred_df)
            
            return render_template('home.html', results=results[0])
            
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080) 