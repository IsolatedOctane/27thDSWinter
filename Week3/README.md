# 강화학습 과제 3주차

발제자: 팀장단

이 과제는 **TODO가 표시된 부분만 구현**하면 됩니다. (그 외 코드는 수정하지 않아도 됩니다.)

### 제출 기한

**2월 1일 (토) 11시 59분**까지

---

## Texas Holdem Agent 만들기

대상 파일: `Train_Q.py` 또는 `Train_SARSA.py` (둘 중 하나 선택)

### 환경/설정
- 환경: PettingZoo의 `texas_holdem_v4`
- 플레이어: `player_0`, `player_1` (2인 대전)
- 학습 알고리즘: **Q-learning** 또는 **SARSA** 중 택 1

### 하이퍼파라미터 (기본값)
| 파라미터 | Q-learning | SARSA |
|----------|------------|-------|
| alpha (학습률) | 0.01 | 0.01 |
| gamma (할인율) | 0.95 | 0.9 |
| epsilon (탐험률) | 0.3 | 0.2 |
| num_episodes | 100 | 100 |

> 하이퍼파라미터 튜닝이 매우 중요합니다. 적어도 base agent보다는 잘해야 합니다.

### 구현할 것 (TODO)

#### Train_Q.py (Q-learning)
- `td_target` 및 `td_error` 계산 (일반 업데이트, 61-62줄)
- `td_target` 및 `td_error` 계산 (터미널 업데이트, 75-76줄)

#### Train_SARSA.py (SARSA)
- `td_target` 및 `td_error` 계산 (일반 업데이트, 69-70줄)
- `td_target` 및 `td_error` 계산 (터미널 업데이트, 87-88줄)

### 실행
```bash
# Q-learning 학습
python Train_Q.py

# SARSA 학습
python Train_SARSA.py

# 테스트 (base agent와 대결)
python Test.py
```


## 제출 방법

1. `Train_Q.py` 또는 `Train_SARSA.py` 중 하나를 선택하여 TODO 부분을 구현하고 학습
2. 학습된 Q-Table이 **pkl 파일**로 자동 저장됨 (`player_0`, `player_1` 각각)
3. `Test.py`로 본인 agent와 base agent를 비교하며 성능 확인
4. 최종 **pkl 파일** (player_0, player_1 둘 다)을 `Agent/` 폴더에 제출


## 의존성
- `requirements.txt` 이 바뀌었으므로 실행하기전 다시 설치해주세요! (Python 3.11 권장)

```bash
# on root dir
pip install -r requirements.txt
```


## 참고사항
- 실제 게임 화면을 보고 싶다면 로컬에서 `Test.py`를 실행하면 됩니다 (pygame 필요)
- pkl 파일을 복제해서 이름만 바꾸는 일은 없도록 해주세요~


## 참고 자료
- [PettingZoo Texas Holdem 문서](https://pettingzoo.farama.org/environments/classic/texas_holdem/)
