# Project Name

Predict Bank Credit Risk using South German Credit Data
predict-bank . #docker
## Overview


This project is designed to predict credit risk using a machine learning model. It includes data ingestion, transformation, model training, prediction, and model monitoring pipelines. The project is implemented using Python and Flask, with a Cassandra database for data storage.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Workflow](#workflow)
5. [Customization](#customization)
6. [Contributing](#contributing)
7. [License](#license)

## Project Structure

```plaintext
├── .github\workflows
│   ├── main.yaml                  # Deployment
├── src/mlproject
│   ├── components                 # Main Coding
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_training.py
│   ├── pipelines/                 # Flow for running the code
│   │   ├── prediction_pipeline.py
│   │   ├── training_pipeline.py  
│   ├── exception.py               # For Custom Exception
│   ├── logger.py                  # Logging file
│   ├── utils.py
├── templates/                     # HTML templates for Flask app
│   ├── history.html                       
│   ├── index.html
│   ├── train.html
│   ├── test.html
├── app.py                         # Main Flask application
├── Dockerfile                     # Docker configuration
├── LICENSE                        # License
├── README.md                      # Project documentation        
├── requirements.txt               # All library list needed to use 
├── setup.py                       # Creating the package
├── template.py                    # Created files/folder with help this file


## Installation


1. Clone the repository:


<pre><code class="language-python">

echo git clone https://github.com/kunalshelke90/Predict-Bank-Credit-Risk-using-South-German-Credit-Data.git

</code></pre>



echo cd Predict-Bank-Credit-Risk-using-South-German-Credit-Data


2. Create a virtual environment and install dependencies:


conda create -p myenv python=3.9 -y

conda activate myenv

pip install -r requirements.txt

3. Set up ".env" environment variables:


Create a .env file in the root directory and add the necessary environment variables:

CASSANDRA_USER = "clientid"
CASSANDRA_PASSWORD = "secret"
CASSANDRA_KEYSPACE = " keyspace name"
CASSANDRA_SECURE_BUNDLE ="your_data_base_name.zip"

## Usage

# Running the Flask Application

1. Start the Flask application:

python app.py

2. Access the application:

Open your web browser and go to http://localhost:8080 to interact with the application.
or http://127.0.0.1:8080

# Using Docker

1. Build the Docker image:

docker build -t predict-bank .

2. Run the Docker container:

docker run -p 5000:5000 predict-bank

3. Access the application:

Open your web browser and go to http://localhost:5000

## Workflow

1. Data Ingestion:

The data is ingested and stored in the artifacts/ folder. This step involves loading data from the Cassandra database , names of the columns and rename to english and target column position is change and set it to last of table .Here data is divided into 2 parts train.csv , test.csv and entire raw data is saved as raw.csv 

2. Data Transformation:

The ingested data is transformed using RobustScaler(keep every feature in same range and handle outliers),SimpleImputer(to deal with missing values) and convert the train and test data into array ,stored in the form preprocessor.pkl in artifacts/preprocessor.pkl

3. Model Training:

The transformed data is used to train machine learning models. The best-performing model is selected based on accuracy and other metrics. Stored the file as model.pkl in artifacts/model

4. Prediction:

The trained model is used to make predictions on new data inputs provided through the test.html page.

## customization

1. Data Ingestion:

Customize data_ingestion.py in the src/mlproject/components folder to suit your data source and schema. You can modify the connection settings for your Cassandra database and adjust the data loading logic in src/mlproject/utils.py .

2. Data Transformation:

Modify data_transformation.py in the src/mlproject/components folder to apply different scaling methods, feature engineering techniques, or transformations according to your dataset's needs.

3. Model Training:

Customize model_training.py in the src/mlproject/components folder to experiment with different models, hyperparameters, and evaluation metrics. You can also integrate other ML libraries like TensorFlow or PyTorch.

4. Web Interface:

Modify the HTML templates in the templates/ folder to match your preferred UI design. You can add or remove input fields, change styles, and customize the prediction output format.


# Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes.

# License

This project is licensed under the MIT License. See the LICENSE file for more details.





## End to End MAchine Learning Project

1. Docker Build checked
2. Github Workflow
3. Iam User In AWS

## Docker Setup In EC2 commands to be Executed

#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

## Configure EC2 as self-hosted runner:

## Setup github secrets:

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo link-  566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = mltest