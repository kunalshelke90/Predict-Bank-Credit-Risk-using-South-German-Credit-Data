import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from dataclasses import dataclass
from src.mlproject.utils import load_object,evalute_models,save_object
import dagshub
import mlflow
from urllib.parse import urlparse
import warnings
import sys

dagshub.init(repo_owner='shelkekunal90', repo_name='Predict-Bank-Credit-Risk-using-South-German-Credit-Data', mlflow=True)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifacts','model.pkl')
    best_model_name=""
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_Config=ModelTrainerConfig()

    def eval_metrics(self, actual, pred, pred_proba=None):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
    
        # If probability predictions are provided, calculate the AUC-ROC
        if pred_proba is not None:
            roc_auc = roc_auc_score(actual, pred_proba, multi_class='ovr', average='weighted')
        else:
            roc_auc = None

            return accuracy, precision, recall, f1, roc_auc
        
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Splitting it into training and test input data')
            
            # X_train,y_train,X_test,y_test=(
            #     train_arr[:,:-1],
            #     train_arr[:,-1],
            #     test_arr[:,:-1],
            #     test_arr[:,-1]
            #     )
            
            # train_arr=np.array(train_arr)
            # test_arr=np.array(test_arr)
            
            X_train=train_arr[:,:-1]
            y_train=train_arr[:,-1]
            X_test=test_arr[:,:-1]
            y_test=test_arr[:,-1]
            
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            
            models = {
                "Logistic Regression": LogisticRegression(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "SVM": SVC(random_state=42),
                "XGBoost": XGBClassifier(random_state=42),
                "CatBoost": CatBoostClassifier(verbose=False,random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                }
            
            params = {
                "Logistic Regression": {
                    'C': [0.1, 1.0, 10.0]
                    },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None]
                    },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [10, 20, None]
                    },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1, 0.05],
                    'n_estimators': [50, 100, 200]
                    },
                "SVM": {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf']
                    },
                "XGBoost": {
                    'learning_rate': [0.01, 0.1, 0.05],
                    'n_estimators': [50, 100, 200]
                    },
                "CatBoost": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.05],
                    'iterations': [30, 50, 100]
                    },
                "AdaBoost": {
                    'learning_rate': [0.01, 0.1, 0.05],
                    'n_estimators': [50, 100, 200],
                    'algorithm': ['SAMME'] 
                    }
                }
            
            model_report:dict=evalute_models(X_train,y_train,X_test,y_test,models,params)
            
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            print("This is the best model:")
            print(best_model_name)
            self.model_trainer_Config.best_model_name=best_model_name
            
            model_names=list(params.keys())
            
            actual_model=""
            
            for model in model_names:
                if best_model_name==model:
                    actual_model=actual_model+model
            
            best_params=params[actual_model]
            
            #mlflow
            
            mlflow.set_registry_uri('https://dagshub.com/shelkekunal90/Predict-Bank-Credit-Risk-using-South-German-Credit-Data.mlflow')
            tracking_url_type_store =urlparse(mlflow.get_tracking_uri()).scheme
            
            with mlflow.start_run(nested=True):
                
                predicted_qualities=best_model.predict(X_test)
                
                (accuracy, precision, recall, f1, roc_auc) = self.eval_metrics(y_test, predicted_qualities)
                
                # mlflow.log_param(best_params)
                for param_name, param_value in best_params.items():
                    mlflow.log_param(param_name, param_value)
                
                mlflow.log_metric("accuracy",accuracy)
                mlflow.log_metric("precision",precision)
                mlflow.log_metric("recall",recall)
                mlflow.log_metric("f1",f1)
                # mlflow.log_metric("roc_auc",roc_auc)
                if roc_auc is not None:
                    mlflow.log_metric("roc_auc", roc_auc)
                    
                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")
            
            
            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            logging.info(f"Best found model for both train and test dataset")
            
            save_object(
                file_path=self.model_trainer_Config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            
            acc_score=accuracy_score(y_test,predicted)
            
            return acc_score , best_model_name
            
        except Exception as e:
            raise CustomException(e,sys)