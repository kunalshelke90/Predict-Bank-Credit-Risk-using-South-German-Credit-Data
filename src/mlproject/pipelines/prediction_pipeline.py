import sys,os
import pandas as pd
from src.mlproject.exception import CustomException
from src.mlproject.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            print('Loading model and preprocessor...')
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            print('Transforming data...')
            data_scaled = preprocessor.transform(features)
            
            print('Predicting...')
            preds = model.predict(data_scaled)
            
            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
        
class CustomData:
    def __init__(
        self,
        status,
        duration: int,
        credit_history,
        purpose,
        amount: int,
        savings,
        employment_duration,
        installment_rate,
        personal_status_sex,
        other_debtors,
        present_residence,
        property,
        age: int,
        other_installment_plans,
        housing,
        number_credits,
        job,
        people_liable,
        telephone,
        foreign_worker,
    ):

        try:
            self.status = status
            self.duration = duration
            self.credit_history = credit_history
            self.purpose =  purpose
            self.amount = amount
            self.savings = savings
            self.employment_duration =  employment_duration
            self.installment_rate =  installment_rate
            self.personal_status_sex = personal_status_sex
            self.other_debtors = other_debtors
            self.present_residence = present_residence
            self.property = property
            self.age = age
            self.other_installment_plans =  other_installment_plans
            self.housing = housing
            self.number_credits = number_credits
            self.job = job
            self.people_liable = people_liable
            self.telephone = telephone
            self.foreign_worker =foreign_worker

        except KeyError as e:
            print(f"Error mapping value: {e}")
            raise CustomException(f"Error mapping value: {e}", sys)

    # def map_value(self, category, value):
    #     try:
    #         mapped_value = CATEGORICAL_MAPPINGS[category][value]
    #         print(f"Mapped {category} ({value}) to {mapped_value}")
    #         return mapped_value
    #     except KeyError as e:
    #         print(f"Error mapping {category} with value '{value}': {e}")
    #         raise CustomException(f"Error mapping {category} with value '{value}': {e}", sys)
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "status": [self.status],
                "duration": [self.duration],
                "credit_history": [self.credit_history],
                "purpose": [self.purpose],
                "amount": [self.amount],
                "savings": [self.savings],
                "employment_duration": [self.employment_duration],
                "installment_rate": [self.installment_rate],
                "personal_status_sex": [self.personal_status_sex],
                "other_debtors": [self.other_debtors],
                "present_residence": [self.present_residence],
                "property": [self.property],
                "age": [self.age],
                "other_installment_plans": [self.other_installment_plans],
                "housing": [self.housing],
                "number_credits": [self.number_credits],
                "job": [self.job],
                "people_liable": [self.people_liable],
                "telephone": [self.telephone],
                "foreign_worker": [self.foreign_worker],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

# class CustomData:
#     def __init__(self,
#                  status: int,
#                  duration: int,
#                  credit_history: int,
#                  purpose: int,
#                  amount: float,
#                  savings: int,
#                  employment_duration: int,
#                  installment_rate: int,
#                  personal_status_sex: int,
#                  other_debtors: int,
#                  present_residence: int,
#                  property: int,
#                  age: int,
#                  other_installment_plans: int,
#                  housing: int,
#                  number_credits: int,
#                  job: int,
#                  people_liable: int,
#                  telephone: int,
#                  foreign_worker: int):
        
#         self.status = status
#         self.duration = duration
#         self.credit_history = credit_history
#         self.purpose = purpose
#         self.amount = amount
#         self.savings = savings
#         self.employment_duration = employment_duration
#         self.installment_rate = installment_rate
#         self.personal_status_sex = personal_status_sex
#         self.other_debtors = other_debtors
#         self.present_residence = present_residence
#         self.property = property
#         self.age = age
#         self.other_installment_plans = other_installment_plans
#         self.housing = housing
#         self.number_credits = number_credits
#         self.job = job
#         self.people_liable = people_liable
#         self.telephone = telephone
#         self.foreign_worker = foreign_worker

#     def get_data_as_data_frame(self):
#         try:
#             custom_data_input_dict = {
#                 "status": [self.status],
#                 "duration": [self.duration],
#                 "credit_history": [self.credit_history],
#                 "purpose": [self.purpose],
#                 "amount": [self.amount],
#                 "savings": [self.savings],
#                 "employment_duration": [self.employment_duration],
#                 "installment_rate": [self.installment_rate],
#                 "personal_status_sex": [self.personal_status_sex],
#                 "other_debtors": [self.other_debtors],
#                 "present_residence": [self.present_residence],
#                 "property": [self.property],
#                 "age": [self.age],
#                 "other_installment_plans": [self.other_installment_plans],
#                 "housing": [self.housing],
#                 "number_credits": [self.number_credits],
#                 "job": [self.job],
#                 "people_liable": [self.people_liable],
#                 "telephone": [self.telephone],
#                 "foreign_worker": [self.foreign_worker],
#             }

#             return pd.DataFrame(custom_data_input_dict)
        
#         except Exception as e:
#             raise CustomException(e, sys)
        
        
            

        # try:
        #     self.status = CATEGORICAL_MAPPINGS['status'][status]
        #     print(f"Mapped status: {self.status}")
            
        #     self.duration = duration
            
        #     self.credit_history = CATEGORICAL_MAPPINGS['credit_history'][credit_history]
        #     print(f"Mapped credit_history: {self.credit_history}")
            
        #     self.purpose = CATEGORICAL_MAPPINGS['purpose'][purpose]
        #     print(f"Mapped purpose: {self.purpose}")
            
        #     self.amount = amount
            
        #     self.savings = CATEGORICAL_MAPPINGS['savings'][savings]
        #     print(f"Mapped savings: {self.savings}")
            
        #     self.employment_duration = CATEGORICAL_MAPPINGS['employment_duration'][employment_duration]
        #     print(f"Mapped employment_duration: {self.employment_duration}")
            
        #     self.installment_rate = CATEGORICAL_MAPPINGS['installment_rate'][installment_rate]
        #     print(f"Mapped installment_rate: {self.installment_rate} ")
            
        #     self.personal_status_sex = CATEGORICAL_MAPPINGS['personal_status_sex'][personal_status_sex]
        #     print(f"Mapped personal_status_sex: {self.personal_status_sex}")
            
        #     self.other_debtors = CATEGORICAL_MAPPINGS['other_debtors'][other_debtors]
        #     print(f"Mapped other_debtors: {self.other_debtors}")
            
        #     self.present_residence = CATEGORICAL_MAPPINGS['present_residence'][present_residence]
        #     print(f"Mapped present_residence: {self.present_residence}")
            
        #     self.property = CATEGORICAL_MAPPINGS['property'][property]
        #     print(f"Mapped property: {self.property}")
            
        #     self.age = age
            
        #     self.other_installment_plans = CATEGORICAL_MAPPINGS['other_installment_plans'][other_installment_plans]
        #     print(f"Mapped other_installment_plans: {self.other_installment_plans}")
            
        #     self.housing = CATEGORICAL_MAPPINGS['housing'][housing]
        #     print(f"Mapped housing: {self.housing}")
            
        #     self.number_credits = CATEGORICAL_MAPPINGS['number_credits'][number_credits]
        #     print(f"Mapped number_credits: {self.number_credits}")
            
        #     self.job = CATEGORICAL_MAPPINGS['job'][job]
        #     print(f"Mapped job: {self.job}")
            
        #     self.people_liable = CATEGORICAL_MAPPINGS['people_liable'][people_liable]
        #     print(f"Mapped people_liable: {self.people_liable}")
            
        #     self.telephone = CATEGORICAL_MAPPINGS['telephone'][telephone]
        #     print(f"Mapped telephone: {self.telephone}")
            
        #     self.foreign_worker = CATEGORICAL_MAPPINGS['foreign_worker'][foreign_worker]
        #     print(f"Mapped foreign_worker: {self.foreign_worker}")

        # except KeyError as e:
        #     print(f"Error mapping value: {e}")
        #     raise CustomException(f"Error mapping value: {e}", sys)
        
        
        # Define mappings for categorical columns
# CATEGORICAL_MAPPINGS = {
#     'status': {'no checking account': 1, '... < 0 DM': 2 ,'0<= ... < 200 DM' :3 ,'... >= 200 DM / salary for at least 1 year':4},
#     'credit_history': {'delay in paying off in the past': 0, 'critical account/other credits elsewhere': 1, 'no credits taken/all credits paid back duly': 2, 'existing credits paid back duly till now': 3, 'all credits at this bank paid back duly': 4},
#     'purpose': {'others': 0 ,'car (new)': 1, 'car (used)': 2, 'furniture/equipment': 3, 'radio/television': 4, 'domestic appliances': 5, 'repairs': 6, 'education': 7, 'vacation': 8, 'retraining': 9, 'business': 10},
#     'savings': {'unknown/no savings account': 1, '... <  100 DM': 2, '100 <= ... <  500 DM': 3, '500 <= ... < 1000 DM': 4, '... >= 1000 DM': 5},
#     'employment_duration': {'unemployed': 1, '< 1 year': 2, '1 <= ... < 4 yrs': 3, '4 <= ... < 7 yrs': 4, '>= 7 yrs': 5},
#     'installment_rate' : {'>= 35' :1, '25 <= ... < 35' :2 ,'20 <= ... < 25' :3 ,'< 20' :4},
#     'personal_status_sex': {'male : divorced/separated': 1, 'female : non-single or male : single': 2, 'male : married/widowed': 3, 'female : single': 4},
#     'other_debtors': {'none': 1, 'co-applicant': 2, 'guarantor': 3},
#     'present_residence':{'< 1 yr' :1 ,'1 <= ... < 4 yrs' :2,'4 <= ... < 7 yrs':3,'>= 7 yrs' :4},
#     'property': {'unknown / no property': 1, 'car or other': 2, 'building society savings': 3, 'real estate': 4},
#     'other_installment_plans': {'bank': 1, 'stores': 2, 'none': 3},
#     'housing': {'for free': 1, 'rent': 2, 'own': 3},
#     'number_credits' : {'1' : 1, '2-3' :2, '4-5':3 , '>=6' :4},
#     'job': {'unemployed/unskilled - non-resident': 1, 'unskilled - resident': 2, 'skilled employee/official ': 3, 'manager/self-employee/highly qualify employee': 4},
#     'people_liable':{'3 or more':1 , '0 to 2' :2},
#     'telephone': {'none': 1, 'yes, registered under the customer name': 2},
#     'foreign_worker': {'yes': 1, 'no': 2}
# }