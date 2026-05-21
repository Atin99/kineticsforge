FROM python:3.11-slim

WORKDIR /app

# Install deploy-only dependencies (no torch, no training libs)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy only what the server needs to run
COPY serve_lite.py .
COPY serve.py .
COPY webapp/ webapp/
COPY api/__init__.py api/__init__.py
COPY api/chat_assistant.py api/chat_assistant.py
COPY core/ core/
COPY data/__init__.py data/__init__.py
COPY data/byod_pipeline.py data/byod_pipeline.py
COPY inference/ inference/
COPY modules/ modules/

# Checkpoints may be empty — the app runs without them
COPY checkpoints/ checkpoints/

# .env is NOT copied — secrets are set via platform env vars
EXPOSE 8000

CMD ["python", "serve.py"]
