FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
# Copy the contents of your project to the container
COPY . /app

# Update package lists and install AWS CLI
RUN apt update -y && apt install awscli -y

# Install the required Python packages
RUN pip install -r requirements.txt

# Expose the port your Flask app is running on (if needed)
EXPOSE 8080

# Command to run your Flask application
CMD ["python3", "app.py"]


