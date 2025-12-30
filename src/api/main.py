"""FDS API 메인 모듈"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    ErrorResponse
)
from .predictor import FDSPredictor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 전역 Predictor
predictor: FDSPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 lifespan 이벤트"""
    # Startup
    global predictor
    logger.info("FDS API 시작...")

    # 모델 경로 (환경변수 또는 기본값)
    models_dir = os.getenv("MODELS_DIR", "models")

    predictor = FDSPredictor()
    try:
        predictor.load_models(models_dir)
        logger.info("모델 로딩 완료!")
    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        # 모델 로딩 실패해도 서버는 시작 (health check에서 확인)

    yield  # 앱 실행

    # Shutdown
    logger.info("FDS API 종료...")


# FastAPI 앱 생성
app = FastAPI(
    title="FDS API",
    description="XGBoost 기반 이상거래 탐지 API (SHAP 설명 포함)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    """루트 엔드포인트"""
    return {
        "message": "FDS API is running!",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """헬스 체크

    서버 상태 및 모델 로딩 여부를 확인합니다.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None and predictor.is_loaded,
        version="1.0.0"
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        500: {"model": ErrorResponse, "description": "서버 오류"}
    },
    tags=["Prediction"]
)
def predict(request: PredictRequest):
    """사기 예측

    거래 정보를 받아 사기 확률과 SHAP 기반 설명을 반환합니다.

    - **transaction_id**: 거래 고유 ID
    - **TransactionAmt**: 거래 금액 (필수, 양수)
    - **ProductCD**: 상품 코드
    - **card1~card6**: 카드 정보
    - **hour**: 거래 시간 (0-23)
    """
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="모델이 로딩되지 않았습니다."
        )

    start_time = time.time()

    try:
        response = predictor.predict(request)
        elapsed = (time.time() - start_time) * 1000

        logger.info(
            f"예측 완료: {request.transaction_id} "
            f"(prob={response.fraud_probability:.4f}, "
            f"fraud={response.is_fraud}, "
            f"time={elapsed:.1f}ms)"
        )

        return response

    except Exception as e:
        logger.error(f"예측 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"예측 중 오류 발생: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=List[PredictResponse],
    tags=["Prediction"]
)
def predict_batch(requests: List[PredictRequest]):
    """배치 사기 예측

    여러 거래를 한 번에 예측합니다.
    """
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="모델이 로딩되지 않았습니다."
        )

    start_time = time.time()

    try:
        responses = predictor.predict_batch(requests)
        elapsed = (time.time() - start_time) * 1000

        logger.info(
            f"배치 예측 완료: {len(requests)}건 (time={elapsed:.1f}ms)"
        )

        return responses

    except Exception as e:
        logger.error(f"배치 예측 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"예측 중 오류 발생: {str(e)}"
        )


# 로컬 실행용
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
