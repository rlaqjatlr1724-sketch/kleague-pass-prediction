# ⚽ K-League Pass Destination Prediction

K리그 경기 데이터를 활용한 패스 도착 위치 예측 AI 모델

> 🏆 2024 K-League Data Challenge  
> **Final Score: 14.5m** (Euclidean Distance)  
> **기간**: 2주  
> **1등과의 격차**: 2m (1등 12.5m)

---

## 📋 프로젝트 개요

축구 경기에서 패스 시퀀스를 분석하여 **최종 패스의 도착 좌표(X, Y)**를 예측하는 딥러닝 프로젝트입니다.

### Problem Definition

| 항목 | 설명 |
|------|------|
| **Input** | 에피소드 내 이벤트 시퀀스 (좌표, 이벤트 타입, 결과, 시간 등) |
| **Output** | 마지막 패스의 도착 좌표 `(end_x, end_y)` |
| **Metric** | 유클리드 거리 (meters) |
| **Field** | FIFA 표준 규격 105m × 68m, 좌→우 공격 방향 정규화 |

### 데이터 구조

```
에피소드 = 공이 라인 밖으로 나가기 전까지의 플레이 시퀀스

├── 이벤트 1: Pass    (start_x, start_y) → (end_x, end_y)
├── 이벤트 2: Carry   (start_x, start_y) → (end_x, end_y)
├── 이벤트 3: Duel    (start_x, start_y) → (end_x, end_y)
├── ...
└── 마지막:   Pass    (start_x, start_y) → (?, ?) ← 예측 대상
```

- **Train**: 15,428 에피소드
- **Test**: 2,414 에피소드
- **이벤트 타입**: 61종 (Pass, Carry, Shot, Tackle 등)

---

## 🏗️ 최종 아키텍처: LSTM + LightGBM Hybrid

```
┌─────────────────────────────────────────────────────────────────┐
│                    Episode Sequence                              │
│              [event_1, event_2, ..., event_T]                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
     ┌────┴────┐                       ┌────┴────┐
     │  LSTM   │                       │  Meta   │
     │ Branch  │                       │Features │
     └────┬────┘                       └────┬────┘
          │                                 │
    ┌─────┴─────┐                    ┌──────┴──────┐
    │ 10D Input │                    │ 48 Features │
    │           │                    │             │
    │ • dx, dy  │                    │ • zone_id   │
    │ • dist    │                    │ • dist_goal │
    │ • angle   │                    │ • lag1_dx   │
    │ • time    │                    │ • seq_stats │
    │ • goal_d  │                    │ • player    │
    │ • score_d │                    │ • team      │
    └─────┬─────┘                    └──────┬──────┘
          │                                 │
    Bi-LSTM (2L, 128H)                      │
    + Simple Attention                      │
          │                                 │
    [256D Embedding] ──────────────────────►│
                                            │
                                    ┌───────┴───────┐
                                    │   LightGBM    │
                                    │  (X model,    │
                                    │   Y model)    │
                                    └───────┬───────┘
                                            │
                                    [delta_x, delta_y]
                                            │
                                    + last_position
                                            │
                                    [end_x, end_y]
```

### 핵심 설계 원칙: 흐름 vs 상황 분리

| Component | 역할 | Input |
|-----------|------|-------|
| **LSTM** | 시퀀스 패턴 학습 | 이동 궤적, 방향 변화, 리듬 |
| **LightGBM** | 상황 컨텍스트 | Zone, 골대 거리, 통계, 선수/팀 정보 |

---

## 📊 Feature Engineering

### LSTM 입력 (10D Continuous Features)

매 타임스텝마다 변화하는 **흐름(flow)** 정보:

```python
[dx, dy, distance, angle, time, 
 dist_to_goal, angle_to_goal, dist_to_own_goal, 
 dist_to_center, score_diff]
```

| Feature | 설명 |
|---------|------|
| `dx, dy` | 이전 위치 대비 이동량 (상대좌표) |
| `distance` | 이동 거리 `√(dx² + dy²)` |
| `angle` | 이동 방향 `atan2(dy, dx) / π` |
| `dist_to_goal` | 상대 골대까지 거리 |
| `angle_to_goal` | 상대 골대 방향 |
| `dist_to_own_goal` | 자기 골대까지 거리 |
| `dist_to_center` | 중앙선까지 거리 |
| `score_diff` | 현재 점수차 (실시간 계산) |

### LightGBM 입력 (48 Meta Features)

현재 상태의 **스냅샷(snapshot)** 정보:

```python
LGBM_FEATURES = [
    # 위치 (3)
    'last_x', 'last_y', 'zone_id',
    
    # 골대 관련 (4)
    'dist_to_goal', 'angle_to_goal', 'goal_open_angle', 'dist_to_own_goal',
    
    # 필드 영역 (6)
    'is_left_side', 'is_center', 'is_right_side',
    'is_final_third', 'is_near_touchline', 'min_dist_to_touchline',
    
    # 시퀀스 통계 (14)
    'seq_length', 'mean_distance', 'std_distance', 'max_distance', 'min_distance',
    'mean_angle', 'std_angle', 'forward_ratio', 'backward_ratio',
    'net_x_movement', 'net_y_movement',
    'recent_dx_mean', 'recent_dy_mean', 'recent_dist_mean',
    
    # Lag-1 (4)
    'lag1_dx', 'lag1_dy', 'lag1_dist', 'lag1_angle',
    
    # 선수/팀 (7)
    'role', 'player_avg_dist', 'player_avg_dx',
    'team_avg_pass_dist', 'team_possession_ratio', 'is_home', 'is_set_piece',
    
    # 매치 컨텍스트 (10)
    'match_phase', 'time_delta', 'episode_time',
    'match_hour', 'time_slot',
    'current_team_rest', 'opp_team_rest', 'rest_diff',
    'cumulative_score_diff', 'is_draw'
]
```

### Zone 분류 (3×3 Grid)

```
         x < 0.33    0.33~0.66    x > 0.66
        ┌───────────┬───────────┬───────────┐
y<0.33  │  Zone 0   │  Zone 1   │  Zone 2   │  (Left)
        ├───────────┼───────────┼───────────┤
0.33~66 │  Zone 3   │  Zone 4   │  Zone 5   │  (Center)
        ├───────────┼───────────┼───────────┤
y>0.66  │  Zone 6   │  Zone 7   │  Zone 8   │  (Right)
        └───────────┴───────────┴───────────┘
         Defensive   Midfield    Attacking
```

---

## 🔧 Model Configuration

### LSTM

```python
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
EMBEDDING_DIM = 16
BATCH_SIZE = 64
LSTM_EPOCHS = 100
LR = 0.001
```

- **Bidirectional**: Yes
- **Attention**: Simple Attention (not Multi-Head)
- **Loss**: Huber Loss (delta=0.15)
- **Optimizer**: Adam (weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau

### LightGBM

```python
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': -1,
    'num_leaves': 127,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'n_estimators': 2000
}
```

---

## 📈 실험 결과

### 성능 추이

| Version | 주요 변경 | Val Distance |
|---------|----------|--------------|
| v1 | Baseline LSTM | ~16.2m |
| v5 | + Attention | ~15.5m |
| v7 | + LightGBM Ensemble | ~15.0m |
| v8 | + Zone Features, Player Stats | ~14.9m |
| v9.2 | + Score Diff, Feature Clean | **14.87m** |

### 학습 로그 (v9.2)

```
Epoch 10:  Val Distance = 15.54m (best: 15.50m)
Epoch 20:  Val Distance = 15.33m (best: 15.20m)
Epoch 30:  Val Distance = 15.16m (best: 15.08m)
Epoch 40:  Val Distance = 14.97m (best: 14.97m)
Epoch 50:  Val Distance = 14.93m (best: 14.93m)
Epoch 60:  Val Distance = 14.93m (best: 14.90m)
Epoch 70:  Val Distance = 14.89m (best: 14.87m)
Epoch 80:  Val Distance = 14.94m (best: 14.87m)
Epoch 90:  Val Distance = 14.92m (best: 14.87m)
Epoch 100: Val Distance = 14.92m (best: 14.87m)

✅ LSTM 학습 완료: Best = 14.87m
```

---

## 🧪 실험 기록

### ✅ 효과가 있었던 것들

| 방법 | 개선폭 | 설명 |
|------|--------|------|
| **상대좌표 변환** | ~1.0m | 절대좌표 → (dx, dy) 변환 |
| **Bidirectional LSTM** | ~0.5m | 양방향 컨텍스트 |
| **Simple Attention** | ~0.3m | Multi-Head보다 단순한 구조가 더 효과적 |
| **LSTM + LGBM 앙상블** | ~0.5m | 시퀀스 + 상황 정보 결합 |
| **Y축 대칭 증강** | ~0.2m | Train만 증강 (Val 누수 방지) |
| **Huber Loss** | ~0.2m | 아웃라이어 강건성 |
| **Zone Features** | ~0.2m | 9-zone 분류 명시적 제공 |
| **Score Diff** | ~0.1m | 실시간 점수차 피처 |

### ❌ 효과가 없었던 것들

| 방법 | 결과 | 분석 |
|------|------|------|
| Multi-Head Attention | 오히려 하락 | Simple이 더 효과적 |
| 더 깊은 LSTM (3-4 layers) | 동일 | 데이터 양 대비 과도한 복잡도 |
| 모델 크기 증가 (256H, 512H) | 동일 | 과적합 위험만 증가 |
| Residual Blocks | 하락 | 과적합 |
| Velocity/Acceleration | 효과 없음 | 축구는 물리 모델과 다름 |
| Kalman Filter | 효과 없음 | 최적 vel_weight = 0 |
| Coordinate Noise 증강 | 하락 | 노이즈만 증가 |
| `total_distance` 피처 | 제거 | `mean_distance`와 중복 |
| `is_weekend` 피처 | 제거 | 영향 미미 |

### 🔄 시도했으나 완성 못한 것들

| 방법 | 목적 |
|------|------|
| MDN (Mixture Density Network) | Center Zone multi-modal 해결 |
| Zone별 전문가 네트워크 (MoE) | 영역별 특화 모델 |
| LoRA Fine-tuning | Zone별 효율적 파인튜닝 |

---

## 🔍 핵심 발견: Center Zone 문제

### Zone별 에러 분석

```
         Defensive    Midfield    Attacking
        ┌───────────┬───────────┬───────────┐
 Left   │   ~13m    │   ~14m    │   ~15m    │
        ├───────────┼───────────┼───────────┤
 Center │  ~20m ❌  │  ~19m ❌  │   ~16m    │
        ├───────────┼───────────┼───────────┤
 Right  │   ~13m    │   ~14m    │   ~15m    │
        └───────────┴───────────┴───────────┘
```

### 원인: Multi-modal Distribution

Center Zone에서는 **모든 방향**으로 패스 가능 → 단일 예측점이 **평균으로 수렴**

```
Center Zone:              Side Zone:
     ↖  ↑  ↗                   ↗
      \ | /                    /
       \|/                    /
    ←───●───→             ●───→
       /|\                    
      / | \                   
     ↙  ↓  ↘                   

예측: 중앙 (모두 틀림)      예측: 우측 (비교적 정확)
```

**해결 시도**: MDN으로 확률 분포 예측 → 시간 부족으로 미완성

---

## 🐛 발견한 버그들

### 1. Multi-Head Attention 오버헤드

```python
# ❌ v8.9 이전: MHA 사용
self.attention = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=4)

# ✅ v8.9 이후: Simple Attention
self.attention = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, 1)
)
```

**결과**: Simple Attention이 더 좋은 성능

### 2. Val 데이터 누수 방지

```python
# ❌ 잘못된 방식: 전체 데이터 증강 후 분할
augmented = augment_y_flip_sequences(all_sequences)
train, val = train_test_split(augmented)  # Val에도 증강 데이터 포함!

# ✅ 올바른 방식: 분할 후 Train만 증강
train_raw, val = train_test_split(sequences)
train = augment_y_flip_sequences(train_raw)  # Val은 원본 유지
```

---

## 📁 프로젝트 구조

```
kleague-pass-prediction/
├── README.md
├── requirements.txt
├── k_league_v9_2_pass_prediction.ipynb  # 최종 노트북
├── configs/
│   └── best_config.yaml
└── src/
    ├── data/
    │   ├── dataset.py
    │   └── preprocessing.py
    ├── models/
    │   └── lstm.py
    └── utils/
        └── metrics.py
```

---

## 🚀 실행 방법

### Google Colab

1. `k_league_v9_2_pass_prediction.ipynb` 업로드
2. GPU 런타임 설정 (L4 권장)
3. 데이터 경로 수정:
   ```python
   BASE_DIR = "/content/drive/MyDrive/your_path"
   ```
4. 순차적 실행

### 필요 라이브러리

```
torch>=2.0.0
lightgbm>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

---

## 💡 배운 점

### 1. Simple > Complex
> Multi-Head Attention보다 Simple Attention이 더 효과적

- 638K 파라미터로 충분
- 복잡한 구조가 항상 좋은 건 아님

### 2. Feature Engineering의 중요성
> 모델 구조보다 좋은 피처가 더 중요

- `score_diff` 추가로 ~0.1m 개선
- 중복 피처 제거 (`total_distance`, `is_weekend`)

### 3. Data Augmentation 주의
> Val 데이터 누수 방지 필수

- Train만 증강, Val은 원본 유지
- Y축 대칭만 효과 있음 (Noise 증강은 역효과)

### 4. 문제의 본질 파악
> Center Zone의 multi-modal 분포가 핵심 난제

- 단일 예측점의 한계
- 확률적 접근 (MDN) 필요

### 5. 데이터의 한계
> 15m 근처에서 plateau

- 상대팀 수비 위치 정보 없음
- 선수 개인 성향 정보 부족

---

## 📝 회고

**2주간의 도전 기록**

- ✅ 16.2m → 14.5m (약 1.7m 개선)
- ✅ LSTM + LightGBM 하이브리드 구조 완성
- ✅ Zone별 에러 분석으로 문제 본질 파악
- ❌ Center Zone 문제 해결 미완성
- ❌ 1등(12.5m) 대비 2m 격차

**핵심 교훈**
> "복잡한 모델보다 좋은 피처와 깔끔한 전처리가 더 중요하다"

---

## 📜 License

MIT License

---

*"The best model is often the simplest one that works."*
