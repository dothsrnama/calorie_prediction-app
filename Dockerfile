FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies (needed for packages like XGBoost/OpenCV if applicable)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Ensure start.sh has executable permissions
RUN chmod +x start.sh

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8501

# Run startup script as entrypoint
CMD ["./start.sh"]
