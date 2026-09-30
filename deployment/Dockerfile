# Container image for the pneumonia detection app.
# Build:  docker build -t pneumonia-app .
# Run:    docker run -p 8501:8501 pneumonia-app
FROM python:3.12-slim

# opencv needs these system libraries even in its headless build
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so Docker caches this layer between rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
