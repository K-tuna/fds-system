"""SHAP Explainer for FDS"""

import numpy as np
import shap
from typing import Dict, List, Any


# 피처 한글 매핑
FEATURE_DESC = {
    # 거래 정보
    'TransactionAmt': '거래금액',
    'TransactionAmt_log': '거래금액(로그)',
    'ProductCD': '상품코드',

    # 카드 정보
    'card1': '카드번호1',
    'card2': '카드번호2',
    'card3': '카드번호3',
    'card4': '카드종류',
    'card5': '카드등급',
    'card6': '카드유형',

    # 주소 정보
    'addr1': '청구지주소',
    'addr2': '배송지주소',
    'dist1': '거리1',
    'dist2': '거리2',

    # 이메일 도메인
    'P_emaildomain': '구매자이메일',
    'R_emaildomain': '수령자이메일',

    # 시간 피처
    'hour': '거래시간',
    'dayofweek': '요일',
    'day': '일자',

    # C 피처 (카운팅)
    'C1': '주소일치횟수',
    'C2': '카드사용횟수',

    # D 피처 (시간차)
    'D1': '이전거래간격',
    'D2': '거래시간차',
    'D3': '카드사용간격',
}


class ShapExplainer:
    """XGBoost 모델용 SHAP Explainer"""

    def __init__(self, model, feature_names: List[str]):
        """
        Args:
            model: XGBoost 모델
            feature_names: 피처 이름 리스트
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def _get_base_value(self) -> float:
        """SHAP expected_value를 스칼라로 반환 (버전 호환)"""
        ev = self.explainer.expected_value
        if hasattr(ev, '__len__'):
            return float(ev[0])
        return float(ev)

    def explain(self, X) -> np.ndarray:
        """
        SHAP 값 계산

        Args:
            X: 입력 데이터 (DataFrame 또는 ndarray)

        Returns:
            SHAP 값 배열 (n_samples, n_features)
        """
        return self.explainer.shap_values(X)

    def get_top_features(
        self, 
        shap_values: np.ndarray, 
        sample_idx: int, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        개별 샘플의 Top K 기여 피처 추출

        Args:
            shap_values: SHAP 값 배열
            sample_idx: 샘플 인덱스
            top_k: 상위 K개

        Returns:
            Top K 피처 정보 리스트
        """
        sample_shap = shap_values[sample_idx]
        importance = np.abs(sample_shap)
        top_indices = np.argsort(importance)[-top_k:][::-1]

        return [
            {
                'feature': self.feature_names[i],
                'feature_kr': FEATURE_DESC.get(self.feature_names[i], self.feature_names[i]),
                'shap_value': float(sample_shap[i]),
                'impact': float(importance[i]),
                'direction': 'increase' if sample_shap[i] > 0 else 'decrease'
            }
            for i in top_indices
        ]

    def to_natural_language(
        self, 
        top_features: List[Dict], 
        max_display: int = 3
    ) -> str:
        """
        Top 피처를 자연어 설명으로 변환

        Args:
            top_features: get_top_features() 결과
            max_display: 표시할 최대 피처 수

        Returns:
            자연어 설명 문자열
        """
        lines = ["[사기 판단 근거]"]

        for f in top_features[:max_display]:
            name_kr = f.get('feature_kr', f['feature'])

            if f['direction'] == 'increase':
                direction_kr = "사기 위험 증가"
            else:
                direction_kr = "사기 위험 감소"

            impact = f['impact']
            if impact > 1.0:
                strength = "매우 높음"
            elif impact > 0.5:
                strength = "높음"
            elif impact > 0.2:
                strength = "중간"
            else:
                strength = "낮음"

            lines.append(f"- {name_kr}: {direction_kr} (영향도 {strength})")

        return "\n".join(lines)

    def create_response(
        self,
        X,
        sample_idx: int,
        threshold: float = 0.5,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        API 응답용 설명 생성

        Args:
            X: 입력 데이터
            sample_idx: 샘플 인덱스
            threshold: 사기 판단 임계값
            top_k: Top K 피처

        Returns:
            API 응답 딕셔너리
        """
        # SHAP 계산
        shap_values = self.explain(X)

        # 예측 확률
        pred_prob = float(self.model.predict_proba(X)[sample_idx, 1])

        # Top K 피처
        top_features = self.get_top_features(shap_values, sample_idx, top_k)

        # 자연어 설명
        explanation_text = self.to_natural_language(top_features)

        return {
            "fraud_probability": pred_prob,
            "is_fraud": bool(pred_prob >= threshold),
            "top_factors": [
                {
                    "feature": f['feature'],
                    "feature_kr": f['feature_kr'],
                    "impact": round(f['impact'], 4),
                    "direction": f['direction']
                }
                for f in top_features
            ],
            "explanation_text": explanation_text
        }
