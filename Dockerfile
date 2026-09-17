FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stockbot/ ./stockbot/
COPY dashboard/ ./dashboard/
COPY pyproject.toml README.md ./

# Writable paths for the SQLite store, logs and model artifacts.
RUN mkdir -p data logs models_store && \
    useradd --create-home --shell /bin/bash stockbot && \
    chown -R stockbot:stockbot /app
USER stockbot

# Paper trading only. Overriding this to false stops the app rather than
# enabling live trading.
ENV ALPACA_PAPER=true \
    ENABLE_PAPER_ORDERS=false

HEALTHCHECK --interval=60s --timeout=15s --start-period=20s --retries=3 \
    CMD python -c "from stockbot.config import load_settings; load_settings(require_credentials=False)"

CMD ["python", "-m", "stockbot.cli", "run", "--train-on-start"]
