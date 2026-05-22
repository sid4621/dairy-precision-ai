# Use a stable Python version optimized for Machine Learning
FROM python:3.10.12-slim

# Set the working directory inside the Hugging Face container
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project files into the container
COPY . .

# Hugging Face strictly requires the server to listen on port 7860
EXPOSE 7860

# Start the FastAPI server on port 7860
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
