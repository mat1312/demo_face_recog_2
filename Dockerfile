# Usa una base di Python con i pacchetti necessari
FROM python:3.10

# Imposta la working directory
WORKDIR /app

# Installa dipendenze di sistema necessarie per dlib
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia i file del progetto
COPY . /app

# Crea un virtual environment per le dipendenze
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Installa le dipendenze
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Esponi la porta per FastAPI
EXPOSE 8000

# Avvia l'app con Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
