"""FDS 예측기 모듈"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd

from .schemas import PredictRequest, PredictResponse, FeatureFactor

logger = logging.getLogger(__name__)

# 범주형 인코딩 매핑 (학습 시 사용된 LabelEncoder 기준)
CATEGORY_MAPPINGS = {
    "ProductCD": {"w": 0, "h": 1, "c": 2, "s": 3, "r": 4},
    "card4": {"visa": 0, "mastercard": 1, "american express": 2, "discover": 3},
    "card6": {"debit": 0, "credit": 1, "charge card": 2, "debit or credit": 3},
    "DeviceType": {"desktop": 0, "mobile": 1},
}

# 문자열 피처 (hash 인코딩)
STRING_FEATURES = {"P_emaildomain", "R_emaildomain", "DeviceInfo"}

# 다단계 위험도 임계값 (최적 threshold 0.18 기준)
# 모델이 0.18 이상이면 "사기일 가능성 있음"으로 판단
RISK_THRESHOLDS = {
    "approve": 0.18,   # 0.00 ~ 0.18: 승인 (사기 가능성 낮음)
    "verify": 0.40,    # 0.18 ~ 0.40: 추가인증 (의심)
    "hold": 0.65,      # 0.40 ~ 0.65: 보류 (높은 의심)
    # 0.65 이상: 차단 (사기 확률 높음)
}


def get_risk_level(prob: float) -> str:
    """확률에 따른 위험도 레벨 반환"""
    if prob < RISK_THRESHOLDS["approve"]:
        return "approve"   # 승인
    elif prob < RISK_THRESHOLDS["verify"]:
        return "verify"    # 추가인증
    elif prob < RISK_THRESHOLDS["hold"]:
        return "hold"      # 보류
    else:
        return "block"     # 차단


class FDSPredictor:
    """FDS 예측기 클래스

    XGBoost 모델을 로딩하고 예측 + SHAP 설명을 생성합니다.
    """

    def __init__(self):
        self.xgb_model = None
        self.threshold: float = 0.5
        self.feature_names: List[str] = []
        self.explainer = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """모델 로딩 여부"""
        return self._is_loaded

    def load_models(self, models_dir: str = "models") -> None:
        """모델 로딩

        Args:
            models_dir: 모델 파일 디렉토리 경로
        """
        models_path = Path(models_dir)

        # XGBoost 모델 로딩
        xgb_path = models_path / "xgb_model.joblib"
        if not xgb_path.exists():
            raise FileNotFoundError(f"XGBoost 모델 파일이 없습니다: {xgb_path}")

        logger.info(f"XGBoost 모델 로딩: {xgb_path}")
        xgb_info = joblib.load(xgb_path)

        self.xgb_model = xgb_info["model"]
        self.threshold = xgb_info.get("optimal_threshold", 0.5)

        # 피처 이름 로딩 (모델에서 또는 별도 저장)
        if "feature_names" in xgb_info:
            self.feature_names = xgb_info["feature_names"]
        else:
            # XGBoost Booster에서 피처 이름 추출
            self.feature_names = self.xgb_model.get_booster().feature_names
            if self.feature_names is None:
                # 피처 이름이 없으면 기본값 사용
                n_features = self.xgb_model.n_features_in_
                self.feature_names = [f"f{i}" for i in range(n_features)]

        logger.info(f"피처 수: {len(self.feature_names)}")
        logger.info(f"최적 Threshold: {self.threshold:.2f}")

        # SHAP Explainer 로딩
        try:
            from src.explainer import ShapExplainer
            self.explainer = ShapExplainer(self.xgb_model, self.feature_names)
            logger.info("SHAP Explainer 로딩 완료")
        except Exception as e:
            logger.warning(f"SHAP Explainer 로딩 실패: {e}")
            self.explainer = None

        self._is_loaded = True
        logger.info("모델 로딩 완료!")

    def _extract_features(self, request: PredictRequest) -> pd.DataFrame:
        """요청에서 피처 추출

        Args:
            request: 예측 요청

        Returns:
            피처 DataFrame (1행)
        """
        # 요청 데이터를 딕셔너리로 변환
        data = request.model_dump(exclude={"transaction_id"})

        # 누락된 피처는 0으로 채움, 범주형은 인코딩
        feature_dict = {}
        for fname in self.feature_names:
            if fname in data and data[fname] is not None:
                val = data[fname]
                # 범주형 인코딩 (매핑 테이블)
                if fname in CATEGORY_MAPPINGS:
                    val = CATEGORY_MAPPINGS[fname].get(str(val).lower(), 0)
                # 문자열 피처는 해시 인코딩
                elif fname in STRING_FEATURES:
                    val = hash(str(val).lower()) % 10000  # 0-9999 범위
                feature_dict[fname] = val
            else:
                feature_dict[fname] = 0  # 기본값

        # DataFrame 생성 (모든 값을 float으로 변환)
        df = pd.DataFrame([feature_dict])
        df = df[self.feature_names].astype(float)  # 피처 순서 보장 + 타입 변환
        return df

    def predict(self, request: PredictRequest) -> PredictResponse:
        """예측 수행

        Args:
            request: 예측 요청

        Returns:
            예측 응답 (확률, 판정, SHAP 설명 포함)
        """
        if not self._is_loaded:
            raise RuntimeError("모델이 로딩되지 않았습니다. load_models()를 먼저 호출하세요.")

        # 1. 피처 추출
        features = self._extract_features(request)

        # 2. 예측
        prob = float(self.xgb_model.predict_proba(features)[0, 1])
        risk_level = get_risk_level(prob)
        is_fraud = risk_level == "block"  # 차단 레벨만 사기로 판정

        # 3. SHAP 설명 생성
        top_factors = []
        explanation_text = ""

        if self.explainer is not None:
            try:
                explanation = self.explainer.create_response(
                    features,
                    sample_idx=0,
                    threshold=self.threshold,
                    top_k=5
                )
                top_factors = [
                    FeatureFactor(**f) for f in explanation["top_factors"]
                ]
                explanation_text = explanation["explanation_text"]
            except Exception as e:
                logger.warning(f"SHAP 설명 생성 실패: {e}")
                explanation_text = "설명을 생성할 수 없습니다."

        # 4. 응답 생성
        return PredictResponse(
            transaction_id=request.transaction_id,
            fraud_probability=prob,
            is_fraud=is_fraud,
            risk_level=risk_level,
            threshold=self.threshold,
            top_factors=top_factors,
            explanation_text=explanation_text
        )

    def predict_batch(
        self,
        requests: List[PredictRequest]
    ) -> List[PredictResponse]:
        """배치 예측

        Args:
            requests: 예측 요청 리스트

        Returns:
            예측 응답 리스트
        """
        return [self.predict(req) for req in requests]

    def predict_from_features(self, features: Dict[str, Any]) -> PredictResponse:
        """이미 전처리된 피처로 예측 (인코딩 변환 없이 바로 사용)

        /samples API에서 반환한 447개 피처를 그대로 사용합니다.
        test_features.csv는 이미 전처리된 상태이므로 추가 변환 불필요.

        Args:
            features: 전처리된 피처 딕셔너리 (447개 피처 + transaction_id + _actual_label)

        Returns:
            예측 응답
        """
        if not self._is_loaded:
            raise RuntimeError("모델이 로딩되지 않았습니다. load_models()를 먼저 호출하세요.")

        # 메타 정보 추출
        transaction_id = features.get("transaction_id", "UNKNOWN")

        # transaction_id, _actual_label 제외한 피처만 추출
        feature_values = {
            k: v for k, v in features.items()
            if k not in ["transaction_id", "_actual_label"]
        }

        # DataFrame 생성 (피처 순서 맞추기)
        df = pd.DataFrame([feature_values])

        # 피처 순서 정렬 (모델 학습 시 순서와 일치)
        # 누락된 피처는 0으로, 추가 피처는 무시
        aligned_features = {}
        for fname in self.feature_names:
            if fname in feature_values:
                aligned_features[fname] = feature_values[fname]
            else:
                aligned_features[fname] = 0.0

        df = pd.DataFrame([aligned_features])
        df = df[self.feature_names].astype(float)

        # 예측
        prob = float(self.xgb_model.predict_proba(df)[0, 1])
        risk_level = get_risk_level(prob)
        is_fraud = risk_level == "block"

        # SHAP 설명 생성
        top_factors = []
        explanation_text = ""

        if self.explainer is not None:
            try:
                explanation = self.explainer.create_response(
                    df,
                    sample_idx=0,
                    threshold=self.threshold,
                    top_k=5
                )
                top_factors = [
                    FeatureFactor(**f) for f in explanation["top_factors"]
                ]
                explanation_text = explanation["explanation_text"]
            except Exception as e:
                logger.warning(f"SHAP 설명 생성 실패: {e}")
                explanation_text = "설명을 생성할 수 없습니다."

        return PredictResponse(
            transaction_id=transaction_id,
            fraud_probability=prob,
            is_fraud=is_fraud,
            risk_level=risk_level,
            threshold=self.threshold,
            top_factors=top_factors,
            explanation_text=explanation_text
        )

    def predict_from_features_batch(
        self,
        features_list: List[Dict[str, Any]]
    ) -> List[PredictResponse]:
        """배치 직접 예측

        Args:
            features_list: 전처리된 피처 딕셔너리 리스트

        Returns:
            예측 응답 리스트
        """
        return [self.predict_from_features(f) for f in features_list]
