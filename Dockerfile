FROM python:3.11-slim

# Install git and compilation tools needed for packages like TextGrad or LCB
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install your dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything into the container
COPY . .

# Explicitly install LiveCodeBench in editable mode within the container
RUN pip install -e ./LiveCodeBench

EXPOSE 5050

# Serves your backend/main.py FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "5050"]