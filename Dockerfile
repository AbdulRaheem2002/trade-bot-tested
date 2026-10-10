FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl procps && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY . .

# Set up non-root user and prepare data directory
RUN useradd -m -U tradebot && \
    mkdir -p /app/data && \
    chown -R tradebot:tradebot /app

USER tradebot

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# By default run the agent
CMD ["python", "-m", "app.main"]
