# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for mysqlclient and other Python packages
RUN apt-get update && apt-get install -y \
    pkg-config \
    default-libmysqlclient-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and necessary folders (models, data, and scripts)
COPY api.py .
COPY eval.py .
COPY evaluator.py .
COPY load_data.py .
COPY mf.py .
COPY phoBERT_content.py .
COPY data /app/data
COPY models /app/models

# Expose port 8086 for the FastAPI app
EXPOSE 8086

# Run the FastAPI app with uvicorn on port 8086
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8086"]
