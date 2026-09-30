# Pneumonia Detection App

Streamlit application serving a transfer-learning model that classifies chest
X-rays as Normal, No Lung Opacity / Not Normal, or Lung Opacity (pneumonia).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run with Docker

```bash
docker build -t pneumonia-app .
docker run -p 8501:8501 pneumonia-app
```

## Run in GitHub Codespaces

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Then open the Ports tab, set port 8501 to **Public**, and use the forwarded URL.

## Disclaimer

Decision-support tool only. Not a diagnostic device. Predictions require
clinical confirmation.
