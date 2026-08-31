import os
import sys

import pandas as pd 
import numpy as np

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor)

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from src.utils import save_object, load_object, evaluate_model, find_best_model

@dataclass
class ModelTrainerConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Starting model training")
            logging.info("Data loaded successfully")
            
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]
            logging.info("Data split successfully")
            
            models = {
                "LinearRegression":LinearRegression(),
                "RandomForestRegressor":RandomForestRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "XGBRegressor":XGBRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "CatBoostRegressor":CatBoostRegressor()
            }
            
            logging.info("Model training started")

            report:dict = evaluate_model(x_train,y_train,x_test,y_test,models)
            
            best_model_name = find_best_model(report)

            best_model = models[best_model_name]

            save_object(file_path='artifacts/best_model.pkl', obj=best_model)

            best_model.fit(x_train, y_train)
            best_model_y_pred = best_model.predict(x_test)

            r2 = r2_score(y_test, best_model_y_pred)
            
            logging.info("Model training completed")
            return r2
            
            
        except Exception as e:
            raise CustomException(e, sys)
  

   

