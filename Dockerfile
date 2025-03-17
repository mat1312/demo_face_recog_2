# Usa un'immagine base con le dipendenze necessarie
FROM python:3.10-slim

# Imposta la working directory
WORKDIR /app

# Installa le dipendenze di sistema necessarie per dlib
RUN apt-get update && apt-get install -y \
    cmake \
    libboost-all-dev \
    libopenblas-dev \
    libx11-dev \
    libatlas-base-dev \
    python3-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Crea un ambiente virtuale
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copia i file nel container
COPY . /app

# Installa le dipendenze Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Espone la porta per Streamlit
EXPOSE 8501

# Avvia l'applicazione Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
