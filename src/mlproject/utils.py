import os,sys
from dotenv import load_dotenv
import pickle
from src.mlproject.exception import CustomException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

def connect_to_cassandra(username: str, password: str, keyspace: str):
    cloud_config = {
        'secure_connect_bundle': 'path_to_secure_connect_bundle.zip'
    }
    auth_provider = PlainTextAuthProvider(username=username, password=password)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect(keyspace)
    return session

def load_data_from_cassandra(session, query: str) -> pd.DataFrame:
    rows = session.execute(query)
    df = pd.DataFrame(list(rows))
    return df

def rename_columns(df: pd.DataFrame, new_column_names: list) -> pd.DataFrame:
    df.columns = new_column_names
    return df

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        
    except Exception as e:
        raise CustomException (e,sys)
    

def evalute_models(X_train,y_train,X_test,y_test,models,param):
    try:
        
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            
            gs = GridSearchCV(model, para, cv=5)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            print(f"y_test[:5]: {y_test[:5]}")
            print(f"y_test_pred[:5]: {y_test_pred[:5]}")

            
            # # If y_test_pred might be probabilities, convert them to binary labels
            # if y_test_pred.ndim > 1 and y_test_pred.shape[1] > 1:
            #     y_test_pred = y_test_pred.argmax(axis=1)  # For multi-class problems
            # elif isinstance(y_test_pred[0], float): 
            #     y_test_pred = (y_test_pred > 0.5).astype(int)  # For binary classification
                
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
        
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException (e,sys)