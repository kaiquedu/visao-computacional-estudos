FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/home/app/.cache/huggingface
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY backend ./backend
COPY frontend ./frontend
RUN useradd --create-home app && mkdir -p /home/app/.cache/huggingface && chown -R app /home/app/.cache
USER app
WORKDIR /app/backend
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120", "wsgi:app"]
