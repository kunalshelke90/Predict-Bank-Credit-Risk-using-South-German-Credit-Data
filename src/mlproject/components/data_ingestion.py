import numpy as np
import pandas as pd
import os ,sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')
    
class DataIngestion:
    
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        
    def initate_data_ingestion(self):
    
        try:    
        #datastaxx code
        
            df=pd.read_csv(os.path.join('notebook','South_German_Credit.csv'))
            logging.info(f"Dataset shape: {df.shape}")
    
            logging.info("Reading completed from database")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
        
            df.to_csv(self.data_ingestion_config.raw_data_path , index=False , header=True)
        
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
        
            logging.info(f"Training set shape: {train_set.shape}")
            logging.info(f"Test set shape: {test_set.shape}")
                
            train_set.to_csv(self.data_ingestion_config.train_data_path , index=False , header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path , index=False , header=True)

            logging.info("Data Ingesiton is Completed")


            return self.data_ingestion_config.train_data_path, self.data_ingestion_config.test_data_path
             
        
        except Exception as e:
            raise CustomException(e,sys)