import numpy as np
import pandas as pd
import os ,sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
# from cassandra.cluster import Cluster
# from cassandra.auth import PlainTextAuthProvider
# from src.mlproject.utils import load_data_from_cassandra,connect_to_cassandra,rename_columns

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
            # # Setup Cassandra connection
            # cloud_config = {
            #     'secure_connect_bundle': '<path_to_secure_connect_bundle.zip>'
            # }
            # auth_provider = PlainTextAuthProvider('client_id', 'client_secret')
            # cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            # session = cluster.connect()

            # # Query to fetch data (replace keyspace_name and table_name with your actual names)
            # query = "SELECT * FROM keyspace_name.table_name"
            # rows = session.execute(query)

            # # Replace <path_to_secure_connect_bundle.zip> with the actual path to your Secure Connect Bundle.
            # # Replace 'client_id' and 'client_secret' with your Astra credentials.
            # # Replace keyspace_name.table_name with the actual keyspace and table name from your Astra database.

            # # Convert rows to a DataFrame
            # df = pd.DataFrame(list(rows))

            # # Original German column names
            # german_columns = df.columns.tolist()

            # # English column names mapping
            # english_columns = ['status', 'duration', 'credit_history', 'purpose', 'amount', 'savings', 
            #                    'employment_duration', 'installment_rate', 'personal_status_sex', 'other_debtors', 
            #                    'present_residence', 'property', 'age', 'other_installment_plans', 'housing', 
            #                    'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'credit_risk']

            # # Rename columns to English
            # df.columns = english_columns
            ####################################################################################################################################

            # logging.info("Connecting to Cassandra database")
            # session = connect_to_cassandra(username, password, keyspace)
            
            # query = "SELECT * FROM keyspace_name.table_name"
            
            # logging.info("Loading data from Cassandra")
            # df = load_data_from_cassandra(session, query)
            
            # logging.info("Renaming columns")
            # english_column_names = ['status', 'duration', 'credit_history', 'purpose', 'amount', 
            #                         'savings', 'employment_duration', 'installment_rate', 
            #                         'personal_status_sex', 'other_debtors', 'present_residence', 
            #                         'property', 'age', 'other_installment_plans', 'housing', 
            #                         'number_credits', 'job', 'people_liable', 'telephone', 
            #                         'foreign_worker', 'credit_risk']
            # df = rename_columns(df, english_column_names)
            
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