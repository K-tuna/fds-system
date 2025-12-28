# CPU vs GPU 가속 비교 실험

## 상황 (Situation)
- Optuna 하이퍼파라미터 튜닝이 CPU로 33분 소요
- RTX 2070 Super (8GB VRAM) GPU 활용 가능
- GPU 가속으로 시간 단축 가능한지 실험

## 가설 (Hypothesis)
- 반복 학습이 많은 Optuna가 GPU 효과가 가장 클 것
- 단일 학습인 Early Stopping은 효과 적을 것

## 실험 (Experiment)

```python
# CPU 버전
XGBClassifier(n_estimators=n, max_depth=6, ...)

# GPU 버전
XGBClassifier(
    n_estimators=n,
    max_depth=6,
    tree_method='hist',
    device='cuda',  # GPU 사용
    ...
)
```

## 결과 (Result)

### 속도 비교
| 방법 | CPU | GPU | 향상 |
|------|-----|-----|------|
| Early Stopping | 20초 | 7초 | **2.7배** |
| CV (3-Fold) | 4.4분 | 2.6분 | 1.7배 |
| Optuna (50 trials) | 33분 | 24분 | 1.4배 |

### 예상과 다른 결과!
- **예상**: Optuna (반복 많음) > CV > Early Stopping
- **실제**: Early Stopping (2.7배) > CV (1.7배) > Optuna (1.4배)

## 원인 분석 (Analysis)

### Early Stopping이 가장 빨라진 이유
```
Early Stopping: 1회 학습 (트리 186개 연속 생성)
→ GPU에 데이터 한 번 로드 후 계속 사용
→ 병렬 처리 효과 극대화
```

### Optuna가 상대적으로 느린 이유
```
Optuna: 50 trials × 다른 파라미터
→ 매 trial마다 새 모델 생성
→ GPU 초기화 오버헤드 누적
→ 순수 학습 시간 외 오버헤드가 많음
```

### 시각화
```
Early Stopping:
  [데이터 로드] → [트리1][트리2]...[트리186] → [완료]
                  ↑ GPU 병렬 처리 효과 큼

Optuna:
  [Trial1: 로드→학습→해제] → [Trial2: 로드→학습→해제] → ...
  ↑ 매번 초기화 오버헤드
```

## 결론

### GPU 효과가 큰 경우
- 단일 학습에서 트리 개수가 많을 때
- 데이터를 한 번 로드 후 계속 사용할 때

### GPU 효과가 작은 경우
- 반복 학습에서 매번 새 모델을 생성할 때
- 학습 시간보다 초기화 오버헤드가 클 때

### 실무 추천
| 상황 | 추천 |
|------|------|
| 빠른 실험 | GPU + Early Stopping (7초) |
| 안정적 튜닝 | GPU + CV (2.6분) ← 추천 |
| 최고 성능 | GPU + Optuna (24분) |

## 면접 포인트

> "CPU로 33분 걸리던 Optuna를 GPU로 24분으로 단축했습니다.
>
> 흥미롭게도 **Early Stopping이 2.7배로 가장 큰 향상**을 보였는데,
> 이는 단일 학습에서 GPU 병렬 처리 효과가 극대화되기 때문입니다.
>
> 반면 Optuna는 1.4배에 그쳤는데,
> 매 trial마다 모델을 새로 생성하면서 GPU 초기화 오버헤드가 누적되었습니다.
>
> 이 실험을 통해 **GPU 가속이 항상 선형적으로 빨라지지 않는다**는 것,
> 그리고 **작업 특성에 따라 효과가 다르다**는 것을 배웠습니다."

## 추가 최적화 가능성

1. **Optuna + Pruning**: 성능 안 나오는 trial 조기 중단
2. **Optuna + Parallel**: n_jobs로 병렬 trial 실행
3. **Warm Start**: 이전 trial 결과를 다음 trial에 활용
