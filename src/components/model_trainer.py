import sys
import os
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split train and test input data")

            # Performing train_test_split
            X_train, y_train, X_test, y_test = (train_array[:,:-1],
                                                train_array[:,-1],
                                                test_array[:,:-1],
                                                test_array[:,-1])
            
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            
            model_report, hyp_model_report = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, 
                                               y_test=y_test, models=models, params=params)
            
            # Best model score from model report
            best_model_score = max(sorted(list(model_report.values())))

            # Best model name (based on best_model_score)
            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]

            best_model = models[best_model_name]

            # Best model score from hoy_model_report
            hyp_best_model_score = max(sorted(list(hyp_model_report.values())))

            # Best model name (based on best_model_score)
            hyp_best_model_name= list(hyp_model_report.keys())[
                list(hyp_model_report.values()).index(hyp_best_model_score)
                ]

            hyp_best_model = models[hyp_best_model_name]

            if best_model_score and hyp_best_model_score < 0.6:
                logging.info("Model Evaluation procedure complete: NO BEST MODEL FOUND.")
                raise CustomException("NO BEST MODEL FOUND")
            
            logging.info(f"Model Evaluation procedure complete: {best_model_name} is the best model.")
            logging.info(f"Model Evaluation procedure complete: {hyp_best_model_name} is the best model after Hyper-parameter Tuning.")

            # Saving the best model --> .pkl file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Saved model.pkl sucessfully")

            y_pred = best_model.predict(X_test)
            hyp_y_pred = hyp_best_model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            hyp_r2 = r2_score(y_test, hyp_y_pred)
        

            return r2, hyp_r2

            

        except Exception as e:
            logging.info("Error occured during Model Training/Evaluation procedure")
            raise CustomException(e, sys)
