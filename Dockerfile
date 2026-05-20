FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY serve_lite.py .
COPY serve.py .
COPY webapp/ webapp/
COPY api/__init__.py api/__init__.py
COPY api/chat_assistant.py api/chat_assistant.py
COPY data/__init__.py data/__init__.py
COPY data/byod_pipeline.py data/byod_pipeline.py
COPY inference/ inference/

# checkpoints/ may be empty locally — copy if present, skip if not
COPY checkpoints/ checkpoints/

EXPOSE 8000

CMD ["python", "serve.py"]
