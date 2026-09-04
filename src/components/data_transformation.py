import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_data_transformation(self):
        try:
            logging.info("Creating the preprocessing pipeline")
            train_path = os.path.join("artifacts","train.csv")
            train_df = pd.read_csv(train_path)
            logging.info("Data loaded successfully")
            
            numerical_columns = ["reading_score", "writing_score"] 
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]


            
            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler()),

                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("cat_encoder",OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor=ColumnTransformer(
                [('num',num_pipeline,numerical_columns),
                ('cat',cat_pipeline,categorical_columns)]
            )
           
            logging.info("Preprocessing pipeline created successfully")
            return (
               preprocessor
               
            )
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path: str, test_path: str):
    
        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")

            preprocessing_obj = self.get_data_transformation()
            
            logging.info("Preprocessing object created successfully")
            logging.info("Separating features and target")
           
            target_column = "math_score"
            
            x_train_df =train_df.drop([target_column], axis=1)
            y_train_df=train_df[target_column]

            x_test_df=test_df.drop([target_column], axis=1)
            y_test_df=test_df[target_column]
            
            logging.info("Separated features and target")
            
            input_feature_train = preprocessing_obj.fit_transform(x_train_df)
            input_feature_test = preprocessing_obj.transform(x_test_df)
            
            logging.info("Data transformation completed")


            train_arr = np.c_[input_feature_train, np.array(y_train_df)]
            test_arr = np.c_[input_feature_test, np.array(y_test_df)]

            save_object(
                file_path=self.transformation_config.preprocessing_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("Preprocessing object saved successfully")
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessing_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)