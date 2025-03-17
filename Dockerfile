FROM python:3.9-slim

# Installa dipendenze di sistema per dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia requirements e installa dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice
COPY . .

# Assicurati che la directory meloni_images esista
RUN mkdir -p meloni_images

# Esponi la porta
EXPOSE 8000

# Comando di avvio
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
