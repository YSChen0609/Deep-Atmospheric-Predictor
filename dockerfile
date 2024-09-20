# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the rest of the application code into the container
COPY . .

# Install the dependencies directly
RUN pip install --no-cache-dir numpy scikit-learn pandas tensorflow

# Command to open a terminal
CMD ["/bin/bash"]
