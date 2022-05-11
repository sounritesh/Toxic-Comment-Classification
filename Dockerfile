
FROM python:3.8.5

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -e .

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]