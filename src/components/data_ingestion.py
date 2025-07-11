import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTranformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import Modeltrainer

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion procedure initiated.")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read raw data as DataFrame.')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-Test split procedure initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion performed sucessfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Error occured during Data Ingestion procedure")
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array = data_transformation.initiate_data_tranformation(train_data, test_data)


    model_trainer = Modeltrainer()
    r2, hyp_r2 = model_trainer.initiate_model_trainer(train_array, test_array)
    print(f"Best model r2 score (without hyper-parameter tuning): {r2*100:.2f}%")
    print(f"Best model r2 score (with hyper-parameter tuning): {hyp_r2*100:.2f}%")

