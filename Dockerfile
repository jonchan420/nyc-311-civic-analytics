# The Streamlit app: the dashboard and Ask 311 in offline mode (no model, no secrets).
#   docker build -t nyc-311-civic-analytics .
#   docker run --rm -p 8080:8080 nyc-311-civic-analytics      # then open http://localhost:8080
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

WORKDIR /app

# dependencies first, so code changes don't reinstall them
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY . .
# config/settings.py creates these folders on import; make them now, while still root
RUN mkdir -p data/raw data/processed models visuals \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8080') + '/_stcore/health', timeout=4)"

# shell form so $PORT (set by Cloud Run and similar hosts) is read at start
CMD ["sh", "-c", "exec streamlit run dashboard/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false"]
