from flask import Flask, render_template, request, redirect, url_for
import os
from src.mlproject.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.components.data_transformation import DataTransformation

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        try:
            # Initialize DataTransformation
            data_transformation = DataTransformation()
            
            # Paths to the CSV files
            train_csv_path = os.path.join('artifacts', 'train.csv')
            test_csv_path = os.path.join('artifacts', 'test.csv')
            
            # Transform the data
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_path=train_csv_path,
                test_path=test_csv_path
            )
            
            app.logger.info(f"shape of array: {train_arr.shape}")
            app.logger.info(f"Shape of array: {test_arr.shape}")
            
            # Initialize ModelTrainer and train the model
            model_trainer = ModelTrainer()
            accuracy, best_model = model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            formatted_accuracy = f"{accuracy * 100:.2f}%"
            status = "Training Completed Successfully"        
    
            return render_template('train.html', model=best_model, score=formatted_accuracy, status=status)
    
        except Exception as e:
        
            return render_template('train.html', score="N/A", model="N/A", status=str(e))       

    else:
        return render_template('train.html', score="N/A", model="N/A", status="")


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('test.html')
    else:
        try:
            # Capture form data
            data = CustomData(
                status=request.form.get('status'),
                duration=int(request.form.get('duration')),
                credit_history=request.form.get('credit_history'),
                purpose=request.form.get('purpose'),
                amount=int(request.form.get('amount')),
                savings=request.form.get('savings'),
                employment_duration=request.form.get('employment_duration'),
                installment_rate=int(request.form.get('installment_rate')),
                personal_status_sex=request.form.get('personal_status_sex'),
                other_debtors=request.form.get('other_debtors'),
                present_residence=int(request.form.get('present_residence')),
                property=request.form.get('property'),
                age=int(request.form.get('age')),
                other_installment_plans=request.form.get('other_installment_plans'),
                housing=request.form.get('housing'),
                number_credits=int(request.form.get('number_credits')),
                job=request.form.get('job'),
                people_liable=int(request.form.get('people_liable')),
                telephone=request.form.get('telephone'),
                foreign_worker=request.form.get('foreign_worker')
            )
            
            # Convert data to DataFrame
            pred_df = data.get_data_as_data_frame()
            print(f"DataFrame: {pred_df}")

            # Instantiate the prediction pipeline
            predict_pipeline = PredictPipeline()

            # Perform prediction
            results = predict_pipeline.predict(pred_df)
            print(f"Prediction: {results}")

            if results[0]==0:
                result= "It is not save to give credit "
            else:
                result="It is save to give credit"
            
            # Return result to the template
            return render_template('test.html', prediction=result)

        except Exception as e:
            error_message = str(e)
            print(f"Error: {error_message}")
            return render_template('test.html', prediction="Error occurred: " + error_message)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
