FROM python:3.10-slim

RUN apt-get update

RUN python -m venv /venv && \
    /venv/bin/python -m pip install --upgrade pip

WORKDIR /app
COPY vision /app/vision
COPY submodules /app/submodules

RUN /venv/bin/python -m pip install --no-cache-dir -r /app/vision/requirements.txt

RUN find . -name "*.egg-info" -exec rm -rf {} + && \
    find . -name "__pycache__" -exec rm -rf {} + && \
    find . -name "*.pyc" -exec rm -rf {} + && \
    find . -name "*.png" -exec rm -rf {} + && \
    find . -name "*.jpeg" -exec rm -rf {} + && \
    find . -name "*.jpg" -exec rm -rf {} + \

ENV PYTHONPATH "${PYTHONPATH}:/app:/app/vision:/app/submodules"

EXPOSE 8080

WORKDIR /app
ENTRYPOINT ["/venv/bin/python", "/app/server/main.py"]
