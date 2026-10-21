FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      libgl1 \
      libglib2.0-0 \
      wget \
      ghostscript && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV XDG_CACHE_HOME=/tmp/.cache
ENV HOME=/tmp
ENV NUMBA_CACHE_DIR=/tmp
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--timeout", "300", "server:app"]
