import os
import sys
import pickle

from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, X_test, y_train, y_test, models, params):
    try:
        model_report = {}
        hyp_model_report = {}
        
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
 
            # Training with best parameters
            model.fit(X_train,y_train)

            # predicitons on training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Model Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            model_report[list(models.keys())[i]] = test_r2
        
        logging.info("Model Training procedure performed sucessfully")
        
        logging.info("Initiating Hyper-parameter Tuning procedure.")
        logging.info('Initiating Best Model Parameter Search using RandomizedSearchCV.')

        # Hyper-parameter Tuning
        for i in range(len(params)):
            
            # Performing Randomized Search
            random_search = RandomizedSearchCV(estimator=model, n_iter=50, cv=3, n_jobs=-1, 
                                            param_distributions=params[model_name], random_state=42, verbose=1)
            
            # Model Training on best parameters
            random_search.fit(X_train, y_train)

            # predicitons on training and testing data
            y_train_pred = random_search.predict(X_train)
            y_test_pred = random_search.predict(X_test)

            # Model Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            hyp_model_report[list(params.keys())[i]] = test_r2

        logging.info("Model Training with Hyper-parameter Tuning performed sucessfully.")

        return model_report, hyp_model_report


    except Exception as e:
        logging.info("Error occured during Model Evaluation procedure.")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            return dill.load(f)
    
    except Exception as e:
        raise CustomException(e, sys)