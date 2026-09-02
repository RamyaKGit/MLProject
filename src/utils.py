import os
import sys
from src.exception import CustomException
from src.logger import logging
import pickle
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str) -> object:
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        logging.info("Evaluating model")
        report = {}

        for model_name, model in models.items():

            para=params[model_name]

            logging.info("fitting {model_name} model")
            if para:
                grid_search = GridSearchCV(model, para, cv=3)
                
                grid_search.fit(X_train, y_train)
                
                model = grid_search.best_estimator_

                model = model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            report[model_name] = {
                "mse": mean_squared_error(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred)
            }
        
        logging.info("Model evaluation completed")
        return report
        
    except Exception as e:
        raise CustomException(e, sys)
    

def find_best_model(report):
        try:
            logging.info("Finding best model")
            best_model_name = max(report, key=lambda x: report[x]["r2"])
            logging.info("Best model found:"+best_model_name)
            if(report[best_model_name].get("r2") < 0.6):
                raise Exception("No best model found")
            return best_model_name

        except Exception as e:
            raise CustomException(e, sys)
    