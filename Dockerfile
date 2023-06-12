# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the application code to the working directory
COPY . .

# Expose the port on which the Flask application will run
EXPOSE 8080

# Set environment variables
ENV GOOGLE_APPLICATION_CREDENTIALS="./credentials.json"

# Run the Flask application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
