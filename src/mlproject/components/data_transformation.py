import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object
from sklearn.preprocessing import OneHotEncoder, StandardScaler,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # def get_data_transformation_object(self):
    #     try:
    #         df1 = pd.read_csv(os.path.join('notebook','South_German_Credit.csv'))
    #         df1.columns = df1.columns.str.strip()
            
    #         categorical_columns = [feature for feature in df1.columns if df1[feature].dtype == 'O']
    #         numerical_columns = [feature for feature in df1.columns if df1[feature].dtype != 'O']

    #         logging.info(f"Identified Categorical Columns: {categorical_columns}")
    #         logging.info(f"Identified Numerical Columns: {numerical_columns}")
            
    #         num_pipeline = Pipeline(steps=[
    #             ("imputer", SimpleImputer(strategy='median')),
    #             ('scaler', StandardScaler())
    #         ])

    #         cat_pipeline = Pipeline(steps=[
    #             ("imputer", SimpleImputer(strategy='most_frequent')),
    #             ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
    #             ('scaler', StandardScaler(with_mean=False))
    #         ])

    #         transformers = []

    #         if len(numerical_columns) > 0:
    #             transformers.append(("num_pipeline", num_pipeline, numerical_columns))

    #         if len(categorical_columns) > 0:
    #             transformers.append(("cat_pipeline", cat_pipeline, categorical_columns))

    #         if not transformers:
    #             raise ValueError("No categorical or numerical columns found in the dataset.")

    #         logging.info(f"Categorical Columns: {categorical_columns}")
    #         logging.info(f"Numerical Columns: {numerical_columns}")

    #         preprocessor = ColumnTransformer(transformers=transformers)

    #         return preprocessor
           
    def get_data_transformer_object(self)->Pipeline:
        
        try:
            
            robust_scaler=RobustScaler()#keep every feature in same range and handle outliers
            simple_imputer=SimpleImputer(strategy='median')#replace missing values as with zero
            preprocessor=Pipeline(steps=[("Imputer",simple_imputer),("Robust_scaler",robust_scaler)])
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test file")
            logging.info(f"Columns in Training DataFrame: {train_df.columns}")
            logging.info(f"Columns in Test DataFrame: {test_df.columns}")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'credit_risk'
            logging.info(f"Target column '{target_column_name}' found in the dataset")
            
            # Check if target column exists
            if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
                raise ValueError(f"Target column '{target_column_name}' not found in the dataset")
            
            #training dataframe
            
            input_feature_train_df = train_df.drop(columns=target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            # testing dataframe
            input_feature_test_df = test_df.drop(columns=target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Input Features Training Columns: {input_feature_train_df.columns}")
            logging.info(f"Input Features Test Columns: {input_feature_test_df.columns}")

            logging.info("Applying Preprocessing on training and test dataframe")

            preprocessor_object=preprocessing_obj.fit(input_feature_train_df)
            
            input_feature_train_arr = preprocessor_object.transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_object.transform(input_feature_test_df)

            logging.info("Transformation successful")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
