# 강화학습 과제 1

발제자: 강성우

이 과제는 **TODO가 표시된 함수/부분만 구현**하면 됩니다. (그 외 코드는 수정하지 않아도 됩니다.)

---

## Part 1) 10-armed Bandit (정상/정지형)

대상 파일: `mab_assignment.py`

### 환경/설정
- Arms: `k = 10`
- 각 run마다 진짜 가치 $q^*(a) \sim \mathcal{N}(0,1)$ 를 새로 뽑고 **run 동안 고정**
- 보상: $R_t \sim \mathcal{N}(q^*(A_t), 1)$
- Runs: `500`, Steps: `2000`

### 구현할 것 (TODO)
- `select_action_eps_greedy(...)`
- `update_sample_average(...)`
- `select_action_ucb(...)`
- `update_constant_step_size(...)`

### 비교 알고리즘 (총 3개)
1. **epsilon-greedy**: `epsilon = 0.1`, **sample-average** 업데이트
2. **optimistic init**: $Q_1(a)=5.0$, `epsilon = 0`, `alpha = 0.1` (constant step-size)
3. **UCB**: `c = 2`

### 실행
```bash
python mab_assignment.py
```
- 3개 알고리즘의 평균 보상 곡선이 한 그래프에 그려지면 완료입니다.

---

## Part 2) FrozenLake(16x16, deterministic) — 선형대수로 Random Policy 평가

대상 파일: `frozen_lake_linear_assignment.py`

### 환경/설정
- FrozenLake `16x16`, `is_slippery=False` (결정론)
- 보상: Goal 도착 `+1`, 그 외 `0`
- 할인율: `gamma = 0.9`
- 정책: 랜덤(상/하/좌/우 균등)

### 해야 할 일 (핵심)
1. Random policy에 대한 전이행렬/보상벡터를 만들어서
2. Bellman expectation equation을 선형대수로 풀어 $V$를 계산하고
3. $V$를 16x16 형태로 출력
4. $V$를 이용해 start(0)에서 goal까지 greedy path를 구해 출력

식은 아래를 사용합니다.
$$(I - \gamma P_\pi) v = R_\pi$$

### 서술형 질문
- (Part 2에서처럼) 선형대수로 연립방정식을 풀어 Bellman 기대 방정식의 해를 구할 수 있습니다.
- 그렇다면 최적(Optimality) 벨만 방정식도 같은 방식으로 선형 연립방정식으로 풀 수 있을까요?
- 가능/불가능하다면 그 이유는 무엇인가요?

> 답안은 [mab_assignment.py](mab_assignment.py) 하단 `A:`에 작성하시면 됩니다.

### 구현할 것 (TODO)
- `build_random_policy_matrices(env)`
- `greedy_path_from_value(env, V, ...)`
- `서술형 질문의 답안 작성`

### 실행 (렌더 옵션)
```bash
# GUI 렌더(추천, 실패하면 ansi로 자동 fallback)
python frozen_lake_linear_assignment.py --render human

# 텍스트 렌더(기본)
python frozen_lake_linear_assignment.py --render ansi

# 렌더 끔
python frozen_lake_linear_assignment.py --render none
```

> 참고: `--render human`은 OS/파이썬 버전/추가 패키지(예: pygame) 환경에 따라 동작이 달라질 수 있습니다.

---

## 의존성
- `requirements.txt` 기준으로 설치합니다. (python 3.11에서 테스트됨)

```bash
# / 에서
pip install -r requirements.txt
```

---

## 제출
- TODO를 채운 파일(들)을 제출합니다.
