FROM python:3.10-slim

WORKDIR /app

ENV WAYMO_CELERY_BASE_URL=redis://redis:6379/
ENV WAYMO_CELERY_DATABASE_NUMBER=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install celery kombu redis python-dotenv

COPY . .