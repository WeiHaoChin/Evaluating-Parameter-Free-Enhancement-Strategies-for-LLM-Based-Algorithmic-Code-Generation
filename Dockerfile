# ==========================================
# STAGE 1: Build & Compile Dependencies
# ==========================================
FROM python:3.11-slim AS builder

# Install compilers needed for package installation
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install python packages cleanly into /root/.local
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install LiveCodeBench in editable mode
# COPY LiveCodeBench ./LiveCodeBench
# RUN pip install --user -e ./LiveCodeBench --no-deps

# ==========================================
# STAGE 2: Clean Runtime Environment (With Data)
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /workspace

# 1. Copy ONLY the finished, compiled Python packages (No build garbage)
COPY --from=builder /usr/local /usr/local
ENV PATH=/root/.local/bin:$PATH

# 2. Copy your actual code and your 8.5 GB data directory straight in
COPY . .

EXPOSE 5050

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "5050", "--no-access-log"]
