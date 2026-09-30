# One image, two entry points.
#
#   Streamlit app (default):
#     docker build -t ai-data-analyst .
#     docker run -p 8501:8501 --env-file .env ai-data-analyst
#
#   HTTP API:
#     docker run -p 8000:8000 --env-file .env \
#       -e AI_ANALYST_API_KEYS=change-me \
#       ai-data-analyst uvicorn api:app --host 0.0.0.0 --port 8000
#
# The .env file is passed at run time and never copied into the image
# (see .dockerignore).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as an unprivileged user, so a compromised process cannot write to
# the application code.
RUN useradd --create-home --uid 10001 analyst \
    && chown -R analyst /app/tools
USER analyst

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)" \
    || python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)" \
    || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
