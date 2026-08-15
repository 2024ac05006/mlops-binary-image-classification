# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Upgrade pip to prevent installation errors
RUN pip install --no-cache-dir --upgrade pip

# 1. Install CPU-only PyTorch directly to save memory and space
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Copy and install the rest of the lightweight serving dependencies
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy application source code and trained model artifact
COPY src/ ./src/
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Start FastAPI app with Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]