FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.0 \
    numpy==1.26.4 \
    pydantic==2.9.0

COPY serve_lite.py .
COPY webapp/ webapp/

EXPOSE 8000

CMD ["python", "serve_lite.py"]
