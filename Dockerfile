# FDS API Dockerfile
FROM python:3.11-slim

# 환경 변수
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODELS_DIR=/app/models

# 작업 디렉토리
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/

# 모델 디렉토리 생성 (볼륨 마운트 대비)
RUN mkdir -p /app/models

# 포트 노출
EXPOSE 8000

# 헬스 체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 실행
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
