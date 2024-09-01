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
           
    def get_data_transformer_object(self)->Pipeline:
        
        try:
        
            robust_scaler=RobustScaler()#keep every feature in same range and handle outliers
            simple_imputer=SimpleImputer(strategy='median')#replace missing values 
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

# import sys
# from src.mlproject.exception import CustomException
# from src.mlproject.logger import logging
# from src.mlproject.utils import save_object
# from sklearn.preprocessing import RobustScaler, FunctionTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.feature_selection import SelectKBest, mutual_info_classif
# from dataclasses import dataclass
# import numpy as np
# import pandas as pd
# import os

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()
           
#     def get_data_transformer_object(self, input_feature_train_df, target_feature_train_df)->Pipeline:
#         try:
#             # Imputation
#             simple_imputer = SimpleImputer(strategy='median')
            
#             # Robust Scaler for outlier handling
#             robust_scaler = RobustScaler()
            
#             # Feature engineering: log transformation on highly skewed features
#             log_transformer = FunctionTransformer(np.log1p, validate=True)
            
#             # SelectKBest based on mutual information
#             select_k_best = SelectKBest(score_func=mutual_info_classif, k=15)
            
#             # Define the pipeline for each type of feature
#             numerical_pipeline = Pipeline(steps=[
#                 ("Imputer", simple_imputer),
#                 ("Log_Transformer", log_transformer),
#                 ("Robust_scaler", robust_scaler),
#                 ("Select_K_Best", select_k_best)
#             ])
            
#             # Apply transformations to all features
#             preprocessor = ColumnTransformer(transformers=[
#                 ("num_pipeline", numerical_pipeline, input_feature_train_df.columns)
#             ])
            
#             # Fit the preprocessor pipeline
#             return preprocessor.fit(input_feature_train_df, target_feature_train_df)

#         except Exception as e:
#             raise CustomException(e, sys)

#     def initiate_data_transformation(self, train_path, test_path):
#         try:
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)

#             logging.info("Reading the train and test file")
#             logging.info(f"Columns in Training DataFrame: {train_df.columns}")
#             logging.info(f"Columns in Test DataFrame: {test_df.columns}")

#             target_column_name = 'credit_risk'
#             logging.info(f"Target column '{target_column_name}' found in the dataset")
            
#             # Check if target column exists
#             if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
#                 raise ValueError(f"Target column '{target_column_name}' not found in the dataset")
            
#             #training dataframe
#             input_feature_train_df = train_df.drop(columns=target_column_name, axis=1)
#             target_feature_train_df = train_df[target_column_name]
            
#             # testing dataframe
#             input_feature_test_df = test_df.drop(columns=target_column_name, axis=1)
#             target_feature_test_df = test_df[target_column_name]

#             logging.info(f"Input Features Training Columns: {input_feature_train_df.columns}")
#             logging.info(f"Input Features Test Columns: {input_feature_test_df.columns}")

#             logging.info("Applying Preprocessing on training and test dataframe")

#             # Pass both input features and target to the preprocessor
#             preprocessor_object = self.get_data_transformer_object(input_feature_train_df, target_feature_train_df)
            
#             input_feature_train_arr = preprocessor_object.transform(input_feature_train_df)
#             input_feature_test_arr = preprocessor_object.transform(input_feature_test_df)

#             logging.info("Transformation successful")

#             train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

#             logging.info("Saved preprocessing object")

#             save_object(
#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessor_object
#             )

#             return (
#                 train_arr,
#                 test_arr,
#                 self.data_transformation_config.preprocessor_obj_file_path
#             )

#         except Exception as e:
#             raise CustomException(e, sys)
