# Base Python image
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the FastAPI app port
EXPOSE 8001

# Start the FastAPI app
CMD ["uvicorn", "mainfile:app", "--host", "0.0.0.0", "--port", "8001"]
