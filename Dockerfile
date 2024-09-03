FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy the contents of your project to the container
COPY . /app

# Update package lists and install AWS CLI
RUN apt update -y && apt install awscli -y

# Install the required Python packages
# RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y && pip install -r requirements.txt

# Command to run your Flask application
CMD ["python3", "app.py"]


