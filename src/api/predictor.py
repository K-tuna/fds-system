"""FDS 예측기 모듈"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd

from .schemas import PredictRequest, PredictResponse, FeatureFactor

logger = logging.getLogger(__name__)


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

        # 누락된 피처는 0 또는 NaN으로 채움
        feature_dict = {}
        for fname in self.feature_names:
            if fname in data and data[fname] is not None:
                feature_dict[fname] = data[fname]
            else:
                feature_dict[fname] = 0  # 기본값

        # DataFrame 생성
        df = pd.DataFrame([feature_dict])
        return df[self.feature_names]  # 피처 순서 보장

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
        is_fraud = prob >= self.threshold

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
