FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml ./
COPY uv.lock ./
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY graders/ ./graders/
COPY server/ ./server/
COPY openenv.yaml ./
COPY inference.py ./
COPY README.md ./

EXPOSE 7860

CMD ["python", "-m", "server.app"]
