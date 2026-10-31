FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# веб-сервер мини-приложений (bot/webapp.py): /jangar/ — «Джангар», /healthz — проверка
EXPOSE 8080

# база и лог фидбэка живут в volume, иначе пропадут при пересборке образа
VOLUME ["/data"]

CMD ["python", "-m", "bot.main"]
