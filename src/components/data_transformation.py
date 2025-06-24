import sys
import os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTranformationConfig()

    def get_data_transformer_object(self):
        """
        This fuction is responsible for Data tranformation of Numnerical and Categorical Features.
        
        """
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(sparse_output=False)),
                ]
            )

            logging.info(f"Numerical Features: {numerical_features}")
            logging.info(f"Categorical Features: {categorical_features}")

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', num_pipeline, numerical_features),
                    ('categorical_pipeline', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.info("Error occured during Data Transformation procedure.")
            raise CustomException(e, sys)
        
    def initiate_data_tranformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data read sucessfully.")

            logging.info("Obtaining pre-processor object")
            pre_processing_obj = self.get_data_transformer_object()

            target_feature_name = 'math_score'
            numerical_features = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_feature_name], axis=1)
            target_feature_train_df = train_df[target_feature_name]

            input_feature_test_df = test_df.drop(columns=[target_feature_name], axis=1)
            target_feature_test_df = test_df[target_feature_name]

            logging.info("Applying pre-processing object on Train and Test DataFrame")

            input_feature_train_arr = pre_processing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = pre_processing_obj.transform(input_feature_test_df)
        
            # Concatinating scaled_input_feature_train_arr and target_feature_train_df (we don't scale test data) --> using np.c_
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
                
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved pre-processing objects.")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=pre_processing_obj
            )

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            raise CustomException(e, sys)
        