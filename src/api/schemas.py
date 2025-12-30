"""FDS API Pydantic 스키마"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """예측 요청 스키마"""

    transaction_id: str = Field(..., description="거래 ID")

    # 거래 정보
    TransactionAmt: float = Field(gt=0, description="거래 금액")
    ProductCD: str = Field(default="W", description="상품 코드")

    # 카드 정보
    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None

    # 주소 정보
    addr1: Optional[float] = None
    addr2: Optional[float] = None

    # 이메일 도메인
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None

    # 시간 정보 (선택)
    hour: Optional[int] = Field(default=None, ge=0, le=23)
    dayofweek: Optional[int] = Field(default=None, ge=0, le=6)

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_001",
                "TransactionAmt": 150.0,
                "ProductCD": "W",
                "card1": 10000,
                "card4": "visa",
                "hour": 14
            }
        }


class FeatureFactor(BaseModel):
    """SHAP 피처 기여도"""

    feature: str = Field(..., description="피처 이름 (영문)")
    feature_kr: str = Field(..., description="피처 이름 (한글)")
    impact: float = Field(..., description="영향도 (절대값)")
    direction: str = Field(..., description="방향 (increase/decrease)")


class PredictResponse(BaseModel):
    """예측 응답 스키마"""

    transaction_id: str = Field(..., description="거래 ID")
    fraud_probability: float = Field(..., ge=0, le=1, description="사기 확률")
    is_fraud: bool = Field(..., description="사기 여부")
    threshold: float = Field(..., description="사용된 임계값")
    top_factors: List[FeatureFactor] = Field(..., description="Top 기여 피처")
    explanation_text: str = Field(..., description="자연어 설명")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_001",
                "fraud_probability": 0.87,
                "is_fraud": True,
                "threshold": 0.18,
                "top_factors": [
                    {
                        "feature": "TransactionAmt",
                        "feature_kr": "거래금액",
                        "impact": 0.25,
                        "direction": "increase"
                    }
                ],
                "explanation_text": "[사기 판단 근거]\n- 거래금액: 사기 위험 증가 (영향도 중간)"
            }
        }


class HealthResponse(BaseModel):
    """헬스 체크 응답"""

    status: str = Field(..., description="서버 상태")
    model_loaded: bool = Field(..., description="모델 로딩 여부")
    version: str = Field(default="1.0.0", description="API 버전")


class ErrorResponse(BaseModel):
    """에러 응답"""

    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(default=None, description="상세 정보")
