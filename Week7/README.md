# 강화학습 과제 7주차

이 과제는 **PPO (Proximal Policy Optimization) 알고리즘**을 구현하는 내용입니다.
`ppo.py`를 실행하고 결과를 분석합니다.

### 제출 기한

**3월 2일 (월) 23시 59분**까지

---

## PPO (Proximal Policy Optimization)

대상 파일: `ppo.py`

### 구현/작성할 것 (TODO)

- `ppo.py`의 TODO가 표시된 부분만 작성합니다. (총 3개)
- `ppo.py` 하단의 **질문 답변**을 작성합니다.

### 학습 결과 조건

- `ppo.py` (CartPole-v1): 최고 Total Reward **400 이상**

그래프에서 기준치를 넘는 구간이 보이도록 학습 결과를 확인합니다.
그래프에 기준선을 **빨간 점선**으로 표시하며, 실행 시 자동 저장됩니다.

### 실행

```bash
python ppo.py
```

---

## 제출 방법

1. TODO 부분 작성 (질문 답변 포함)
2. 코드 실행
3. 결과 그래프 이미지를 Week7 폴더에 저장 (저장 코드 구현돼 있음)
   (예: `ppo.png`)
4. 기준 Reward 충족 여부 확인

---

## 실행 환경 세팅

### 가상환경 활성화 (venv, conda 자유)

(예시)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 필수 패키지 설치

Week7 폴더 안의 `requirements.txt`를 사용해서 설치합니다.

```bash
# Week7 폴더에서 실행
pip install -r requirements.txt
```

이미 다른 버전이 설치되어 있어서 에러가 나는 경우 아래처럼 다시 설치합니다.

```bash
# Week7 폴더에서 실행
pip uninstall -y numpy gym gymnasium
pip install -r requirements.txt
```

PyTorch는 공식 사이트(https://pytorch.org)에서 환경에 맞게 설치합니다.

---

## 알고리즘 개요

### PPO (Proximal Policy Optimization)

- Actor-Critic 기반 on-policy 알고리즘
- **Clipped Surrogate Objective**: 정책 업데이트 크기를 제한하여 학습 안정성 확보
- **GAE (Generalized Advantage Estimation)**: TD error의 지수가중 합으로 advantage 추정
- **Multi-epoch 업데이트**: 같은 배치 데이터로 여러 번 학습하여 샘플 효율성 향상
- A2C 대비 학습이 안정적이고, TRPO 대비 구현이 간단

### 핵심 수식

**Clipped Surrogate Objective:**

```
L_clip = min(r(theta) * A, clip(r(theta), 1-eps, 1+eps) * A)
r(theta) = pi_new(a|s) / pi_old(a|s)  (확률 비율)
```

**GAE (Generalized Advantage Estimation):**

```
A_t = delta_t + (gamma * lambda) * delta_{t+1} + (gamma * lambda)^2 * delta_{t+2} + ...
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)  (TD error)
```

---

## A2C -> PPO 발전 과정

| 항목 | A2C (Week6) | PPO (Week7) |
|------|-------------|-------------|
| **업데이트 단위** | 매 스텝 (1-step) | 배치 (N 에피소드) |
| **Advantage 추정** | 1-step TD | GAE (다단계 TD의 가중합) |
| **정책 제약** | 없음 | Clipping (비율 제한) |
| **같은 데이터 재사용** | 불가 (on-policy) | 가능 (여러 에포크) |
| **학습 안정성** | 보통 | 높음 |

---

## 참고 자료

- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
- 밑시딥4권 chapter 9, 10
