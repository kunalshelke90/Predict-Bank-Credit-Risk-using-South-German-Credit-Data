# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the application runs on
EXPOSE 5000

# Define environment variables (if needed)
# ENV FLASK_ENV=production

# Command to run the application
CMD ["python", "app.py"]



# Explanation:
    
#     Base Image: FROM python:3.12-slim uses a lightweight Python 3.12 image.
#     Working Directory: WORKDIR /app sets /app as the working directory within the container.
#     Copy Dependencies: COPY requirements.txt . copies the requirements.txt file to the container.
#     Install Dependencies: RUN pip install --no-cache-dir -r requirements.txt installs the dependencies listed in requirements.txt.
#     Copy Application Code: COPY . . copies all your project files into the container.
#     Expose Port: EXPOSE 5000 tells Docker to expose port 5000 (or whatever port your Flask app is using).
#     Run the Application: CMD ["python", "app.py"] sets the command to run your Flask application when the container starts.
#     Additional Considerations:
#     Environment Variables: If your application requires environment variables, you can add them using ENV.
#     Port Configuration: Ensure the port in EXPOSE matches the port your Flask app uses.
#     Optimizations: You can further optimize the Dockerfile by using multi-stage builds or minimizing the size of the final image.


#Build and Run:
#     Once the Dockerfile is ready, you can build and run your Docker container:

# # Build the Docker image
# docker build -t my-flask-app .

# # Run the Docker container
# docker run -p 5000:5000 my-flask-app

