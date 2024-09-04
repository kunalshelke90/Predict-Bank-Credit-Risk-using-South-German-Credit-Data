FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy the contents of your project to the container
COPY . /app

# Update package lists and install AWS CLI
RUN apt update -y && apt install awscli -y

# Install the required Python packages
RUN pip install -r requirements.txt
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y && pip install -r requirements.txt

# Set environment variables (not recommended for sensitive information)
ENV CASSANDRA_USER=${CASSANDRA_USER}
ENV CASSANDRA_PASSWORD=${CASSANDRA_PASSWORD}
ENV CASSANDRA_KEYSPACE=${CASSANDRA_KEYSPACE}
ENV CASSANDRA_SECURE_BUNDLE=${CASSANDRA_SECURE_BUNDLE}
ENV DAGSHUB_REPO_OWNER=${DAGSHUB_REPO_OWNER}
ENV DAGSHUB_REPO_NAME=${DAGSHUB_REPO_NAME}
ENV DAGSHUB_MLFLOW=${DAGSHUB_MLFLOW}
ENV MLFLOW_REGISTRY_URI=${MLFLOW_REGISTRY_URI}

# Command to run your Flask application
CMD ["python3", "app.py"]
