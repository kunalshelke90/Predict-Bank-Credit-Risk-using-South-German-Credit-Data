# Project Name

Predict Bank Credit Risk using South German Credit Data


## Motivation

Normally, most of the bank's wealth is obtained from providing credit loans so that a marketing bank must be able to reduce the risk of non-performing credit loans. The risk of providing loans can be minimized by studying patterns from existing lending data. One technique that you can use to solve this problem is to use data mining techniques. Data mining makes it possible to find hidden information from large data sets by way of classification.

The goal of this project, you have to build a model to predict whether the person, described by the attributes of the dataset, is a safe (1) or a not safe (0) credit risk

## Dataset Link

https://archive.ics.uci.edu/dataset/522/south+german+credit 

## Overview

This project is designed to predict credit risk using a machine learning model. It includes data ingestion, transformation, model training,and prediction. The project is implemented using Python and Flask, with a Cassandra database for data storage.

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
├── artifacts                      # saved all outputs
├── Documents                      # Documentation
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
```

## Installation


1. Clone the repository:


<pre><code class="language-python">
git clone https://github.com/kunalshelke90/Predict-Bank-Credit-Risk-using-South-German-Credit-Data.git
</code></pre>

<pre><code class="language-python">
cd Predict-Bank-Credit-Risk-using-South-German-Credit-Data
</code></pre>

2. Create a virtual environment and install dependencies:

<pre><code class="language-python">
conda create -p myenv python=3.9 -y
</code></pre>

<pre><code class="language-python">
conda activate myenv
</code></pre>

<pre><code class="language-python">
pip install -r requirements.txt
</code></pre>

3. Set up ".env" environment variables:


Create a .env file in the root directory and add the necessary environment variables:

<pre><code class="language-python">
CASSANDRA_USER = "clientid"
CASSANDRA_PASSWORD = "secret"
CASSANDRA_KEYSPACE = "keyspace name"
CASSANDRA_SECURE_BUNDLE ="your_data_base_name.zip"

DAGSHUB_REPO_OWNER="owner_name"
DAGSHUB_REPO_NAME="Repo_name"
DAGSHUB_MLFLOW="True"
MLFLOW_REGISTRY_URI="https://dagshub.com/repo_owner/repo_name.mlflow"

AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION = 
AWS_ECR_LOGIN_URI = 
ECR_REPOSITORY_NAME = 
</code></pre>


## Usage

# Running the Flask Application

1. Start the Flask application:

<pre><code class="language-python">
python app.py
</code></pre>

2. Access the application:

Open your web browser and go to http://localhost:8080 to interact with the application.
or http://127.0.0.1:8080

# Using Docker

1. Build the Docker image:

<pre><code class="language-python">
docker build -t predict-bank .
</code></pre>

2. Run the Docker container:

<pre><code class="language-python">
docker run -p 5000:5000 predict-bank
</code></pre>

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