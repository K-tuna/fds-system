"""FDS 예측기 모듈 - 스태킹 앙상블 (XGBoost + LightGBM + CatBoost)"""

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

# 다단계 위험도 임계값 (계층형 대응)
# 스태킹 모델 기준으로 조정
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

    스태킹 앙상블 (XGBoost + LightGBM + CatBoost) + SHAP 설명을 생성합니다.
    """

    def __init__(self):
        # 스태킹 모델들
        self.xgb_model = None
        self.lgbm_model = None
        self.cat_model = None
        self.meta_model = None

        self.threshold: float = 0.5
        self.feature_names: List[str] = []
        self.explainer = None
        self._is_loaded = False
        self._use_stacking = False  # 스태킹 사용 여부

    @property
    def is_loaded(self) -> bool:
        """모델 로딩 여부"""
        return self._is_loaded

    def load_models(self, models_dir: str = "models") -> None:
        """모델 로딩

        스태킹 모델이 있으면 스태킹 사용, 없으면 기존 XGBoost 단독 사용

        Args:
            models_dir: 모델 파일 디렉토리 경로
        """
        models_path = Path(models_dir)

        # 스태킹 모델 파일들
        stacking_files = {
            "xgb": models_path / "stacking_xgb_tuned.joblib",
            "lgbm": models_path / "stacking_lgbm_tuned.joblib",
            "cat": models_path / "stacking_cat_tuned.joblib",
            "meta": models_path / "stacking_meta_model.joblib",
            "features": models_path / "stacking_feature_names.joblib",
            "config": models_path / "stacking_config.joblib",
        }

        # 스태킹 모델 존재 여부 확인
        stacking_available = all(f.exists() for f in stacking_files.values())

        if stacking_available:
            self._load_stacking_models(models_path, stacking_files)
        else:
            self._load_single_model(models_path)

        # SHAP Explainer 로딩 (XGBoost 기반)
        self._load_shap_explainer()

        self._is_loaded = True
        model_type = "스태킹 앙상블" if self._use_stacking else "XGBoost 단독"
        logger.info(f"모델 로딩 완료! ({model_type})")

    def _load_stacking_models(self, models_path: Path, files: Dict) -> None:
        """스태킹 모델들 로딩"""
        logger.info("스태킹 앙상블 모델 로딩 중...")

        # 3개 Base 모델 로딩
        self.xgb_model = joblib.load(files["xgb"])
        logger.info(f"  XGBoost 로딩: {files['xgb']}")

        self.lgbm_model = joblib.load(files["lgbm"])
        logger.info(f"  LightGBM 로딩: {files['lgbm']}")

        self.cat_model = joblib.load(files["cat"])
        logger.info(f"  CatBoost 로딩: {files['cat']}")

        # 메타 모델 로딩
        self.meta_model = joblib.load(files["meta"])
        logger.info(f"  Meta 모델 로딩: {files['meta']}")

        # 피처 이름 로딩
        self.feature_names = joblib.load(files["features"])
        logger.info(f"  피처 수: {len(self.feature_names)}")

        # 설정 로딩
        config = joblib.load(files["config"])
        self.threshold = config.get("threshold_methods", {}).get(
            "fpr_constraint", {}
        ).get("threshold", 0.18)
        logger.info(f"  Threshold: {self.threshold:.3f}")

        self._use_stacking = True

    def _load_single_model(self, models_path: Path) -> None:
        """기존 XGBoost 단독 모델 로딩 (fallback)"""
        xgb_path = models_path / "xgb_model.joblib"
        if not xgb_path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {xgb_path}")

        logger.info(f"XGBoost 단독 모델 로딩: {xgb_path}")
        xgb_info = joblib.load(xgb_path)

        self.xgb_model = xgb_info["model"]
        self.threshold = xgb_info.get("optimal_threshold", 0.5)

        # 피처 이름 로딩
        if "feature_names" in xgb_info:
            self.feature_names = xgb_info["feature_names"]
        else:
            self.feature_names = self.xgb_model.get_booster().feature_names
            if self.feature_names is None:
                n_features = self.xgb_model.n_features_in_
                self.feature_names = [f"f{i}" for i in range(n_features)]

        logger.info(f"피처 수: {len(self.feature_names)}")
        logger.info(f"최적 Threshold: {self.threshold:.2f}")

        self._use_stacking = False

    def _load_shap_explainer(self) -> None:
        """SHAP Explainer 로딩 (XGBoost 기반)"""
        try:
            from src.explainer import ShapExplainer
            self.explainer = ShapExplainer(self.xgb_model, self.feature_names)
            logger.info("SHAP Explainer 로딩 완료")
        except Exception as e:
            logger.warning(f"SHAP Explainer 로딩 실패: {e}")
            self.explainer = None

    def _predict_proba(self, features: pd.DataFrame) -> float:
        """예측 확률 반환 (스태킹 또는 단독)"""
        if self._use_stacking:
            # 3개 모델의 예측 확률
            prob_xgb = self.xgb_model.predict_proba(features)[:, 1]
            prob_lgbm = self.lgbm_model.predict_proba(features)[:, 1]
            prob_cat = self.cat_model.predict_proba(features)[:, 1]

            # 메타 피처 생성
            meta_features = np.column_stack([prob_xgb, prob_lgbm, prob_cat])

            # 메타 모델로 최종 예측
            prob = self.meta_model.predict_proba(meta_features)[0, 1]
        else:
            # XGBoost 단독
            prob = self.xgb_model.predict_proba(features)[0, 1]

        return float(prob)

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

        # 2. 예측 (스태킹 또는 단독)
        prob = self._predict_proba(features)
        risk_level = get_risk_level(prob)
        is_fraud = risk_level == "block"  # 차단 레벨만 사기로 판정

        # 3. SHAP 설명 생성 (XGBoost 기반)
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

        # 피처 순서 정렬 (모델 학습 시 순서와 일치)
        aligned_features = {}
        for fname in self.feature_names:
            if fname in feature_values:
                aligned_features[fname] = feature_values[fname]
            else:
                aligned_features[fname] = 0.0

        df = pd.DataFrame([aligned_features])
        df = df[self.feature_names].astype(float)

        # 예측 (스태킹 또는 단독)
        prob = self._predict_proba(df)
        risk_level = get_risk_level(prob)
        is_fraud = risk_level == "block"

        # SHAP 설명 생성 (XGBoost 기반)
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
