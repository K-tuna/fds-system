# tests/test_model.py
"""
FDS 모델 테스트
- 모델 파일 존재 확인
- 기본 추론 테스트
- 비용 함수 테스트
"""

import pytest
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix

# 프로젝트 루트 기준 경로
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_model.joblib"


class TestModel:
    """FDS 모델 테스트"""

    def test_model_file_exists(self):
        """모델 파일 존재 확인"""
        assert MODEL_PATH.exists(), f"모델 파일 없음: {MODEL_PATH}"

    def test_model_can_load(self):
        """모델 로드 가능 확인"""
        if not MODEL_PATH.exists():
            pytest.skip("모델 파일 없음")

        model = joblib.load(MODEL_PATH)
        assert model is not None, "모델 로드 실패"

    def test_model_prediction(self):
        """기본 추론 테스트"""
        if not MODEL_PATH.exists():
            pytest.skip("모델 파일 없음")

        model_data = joblib.load(MODEL_PATH)

        # dict 형태로 저장된 경우 처리
        if isinstance(model_data, dict):
            model = model_data['model']
            n_features = model_data.get('n_features', model.n_features_in_)
        else:
            model = model_data
            n_features = model.n_features_in_

        # 더미 데이터 생성
        X_dummy = np.random.rand(10, n_features)

        # 예측
        proba = model.predict_proba(X_dummy)

        # 검증
        assert proba.shape == (10, 2), f"출력 shape 오류: {proba.shape}"
        assert np.all((proba >= 0) & (proba <= 1)), "확률 범위 오류"

    def test_model_prediction_shape(self):
        """예측 결과 shape 테스트"""
        if not MODEL_PATH.exists():
            pytest.skip("모델 파일 없음")

        model_data = joblib.load(MODEL_PATH)

        # dict 형태로 저장된 경우 처리
        if isinstance(model_data, dict):
            model = model_data['model']
            n_features = model_data.get('n_features', model.n_features_in_)
        else:
            model = model_data
            n_features = model.n_features_in_

        # 단일 샘플
        X_single = np.random.rand(1, n_features)
        proba_single = model.predict_proba(X_single)
        assert proba_single.shape == (1, 2), "단일 샘플 shape 오류"

        # 다중 샘플
        X_batch = np.random.rand(100, n_features)
        proba_batch = model.predict_proba(X_batch)
        assert proba_batch.shape == (100, 2), "다중 샘플 shape 오류"


class TestCostFunction:
    """비용 함수 테스트"""

    def test_cost_calculation_basic(self):
        """기본 비용 계산 테스트"""
        # 간단한 예시: 5건 중 1건 FN
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.6, 0.7, 0.3])  # 마지막 1이 0.3으로 FN
        threshold = 0.5

        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # 검증
        assert fn == 1, f"FN 계산 오류: expected 1, got {fn}"
        assert fp == 0, f"FP 계산 오류: expected 0, got {fp}"
        assert tp == 2, f"TP 계산 오류: expected 2, got {tp}"
        assert tn == 2, f"TN 계산 오류: expected 2, got {tn}"

    def test_cost_calculation_values(self):
        """비용 계산 값 테스트"""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.6, 0.7, 0.3])
        threshold = 0.5
        fn_cost = 1_000_000
        fp_cost = 50_000

        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_fn_cost = fn * fn_cost
        total_fp_cost = fp * fp_cost
        total_cost = total_fn_cost + total_fp_cost

        # 1건 FN × 100만원 = 100만원
        assert total_fn_cost == 1_000_000, f"FN 비용 오류: {total_fn_cost}"
        # 0건 FP × 5만원 = 0원
        assert total_fp_cost == 0, f"FP 비용 오류: {total_fp_cost}"
        # 총 비용 = 100만원
        assert total_cost == 1_000_000, f"총 비용 오류: {total_cost}"

    def test_cost_with_fp(self):
        """FP 포함 비용 테스트"""
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.6, 0.1, 0.1, 0.8, 0.9])  # 첫번째 0이 0.6으로 FP
        threshold = 0.5
        fn_cost = 1_000_000
        fp_cost = 50_000

        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_cost = fn * fn_cost + fp * fp_cost

        # 0건 FN + 1건 FP = 5만원
        assert fn == 0, f"FN 오류: {fn}"
        assert fp == 1, f"FP 오류: {fp}"
        assert total_cost == 50_000, f"총 비용 오류: {total_cost}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
