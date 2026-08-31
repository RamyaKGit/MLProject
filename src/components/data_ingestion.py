import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import sklearn 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")
            data = pd.read_csv("notebook/data/stud.csv")
            logging.info("Data loaded successfully")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.train_data_path, index=False)
            logging.info("Data saved successfully")
            
            train,test = train_test_split(data,test_size=0.2,random_state=42)
            
            train.to_csv(self.ingestion_config.train_data_path, index=False)
            test.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data split successfully")
            
            return (
                str(self.ingestion_config.train_data_path),
                str(self.ingestion_config.test_data_path)
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()
        
        data_transformation = DataTransformation()
        train_arr,test_arr,preprocessing_obj_file_path = data_transformation.initiate_data_transformation(train_path=train_data, test_path=test_data)
        
        logging.info("Initiating model training")

        model_trainer = ModelTrainer()
        logging.info("Best model r2 score="+str(model_trainer.initiate_model_training(train_arr,test_arr)))
    except Exception as e:
        logging.info(e)
        raise CustomException(e, sys)



