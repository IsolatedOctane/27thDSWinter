"""[과제 1 - Part 2] FrozenLake(16x16, deterministic) - 선형대수로 Random Policy 평가 템플릿

요구사항(요약)
- 16x16 FrozenLake, is_slippery=False (결정론)
- 보상: Goal 도착 +1, 나머지 0
- gamma=0.9
- 정책: 랜덤(상/하/좌/우 각 0.25)
- Bellman expectation equation:
    (I - gamma P_pi) v = R_pi
- numpy로 solve/inv로 v 계산 후 16x16 출력
- v 기반 greedy path: start(0)에서 이웃 중 v가 가장 큰 곳으로 이동, goal(nS-1)까지

이번 버전은 `gymnasium`의 FrozenLake 환경을 사용합니다.

주의: 핵심 행렬 구축/경로 로직은 TODO로 비워두었습니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import argparse
import time

import numpy as np

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


@dataclass(frozen=True)
class FrozenLakeSpec:
    gamma: float = 0.9


def make_env(
    *,
    size: int = 16,
    p: float = 0.8,
    seed: int = 0,
    render_mode: str | None = None,
) -> gym.Env:
    """Create deterministic NxN FrozenLake env (default 16x16)."""
    desc = generate_random_map(size=size, p=p, seed=seed)
    return gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=False,
        render_mode=render_mode,
    )


def build_random_policy_matrices(env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
    """TODO: build P_pi (nSxnS), R_pi (nS,) for random policy using gymnasium transitions.

        Random policy: pi(a|s)=0.25 for all actions.
        Gymnasium FrozenLake provides transitions as:
            P[s][a] = list of (prob, next_state, reward, terminated)

        - 행렬 차원은 observation_space/action_space 기반으로 일반화
    """
    P = env.unwrapped.P  # type: ignore[attr-defined]
    nS = env.observation_space.n
    nA = env.action_space.n
    pi = 1.0 / nA

    P_pi = np.zeros((nS, nS), dtype=float)
    R_pi = np.zeros((nS,), dtype=float)

    # TODO: fill P_pi and R_pi using P
    raise NotImplementedError


def solve_value_function(P_pi: np.ndarray, R_pi: np.ndarray, gamma: float) -> np.ndarray:
    """Solve (I - gamma P) V = R."""
    nS = P_pi.shape[0]
    I = np.eye(nS, dtype=float)
    A = I - gamma * P_pi
    b = R_pi

    # NOTE: 여기 자체는 핵심이 아니라서 제공(선형대수 solve)
    V = np.linalg.solve(A, b)
    return V


def greedy_path_from_value(
    env: gym.Env,
    V: np.ndarray,
    *,
    start: int = 0,
    goal: int | None = None,
    max_steps: int = 1000,
) -> list[int]:
    """TODO: Greedy path by choosing action that maximizes V[s_next].
    """
    # TODO: implement using env.unwrapped.P
    raise NotImplementedError


def pretty_print_value(V: np.ndarray, env: gym.Env) -> None:
    nrow = env.unwrapped.nrow  # type: ignore[attr-defined]
    ncol = env.unwrapped.ncol  # type: ignore[attr-defined]
    grid = V.reshape(nrow, ncol)
    with np.printoptions(precision=3, suppress=True):
        print(grid)


def render_greedy_rollout(env: gym.Env, path: list[int], *, delay_sec: float = 0.1) -> None:
    """Render the greedy path (best-effort)."""
    if env.render_mode is None:
        return
    try:
        env.reset()
        if env.render_mode == "ansi":
            print(env.render())
        for next_state in path[1:]:
            P = env.unwrapped.P  # type: ignore[attr-defined]
            cur_s = int(getattr(env.unwrapped, "s", path[0]))  # type: ignore[attr-defined]
            chosen_a = None
            for a in range(env.action_space.n):
                tr = P[cur_s][a]
                _, s2, _, _ = max(tr, key=lambda x: x[0])
                if int(s2) == int(next_state):
                    chosen_a = a
                    break
            if chosen_a is None:
                break
            env.step(chosen_a)
            if env.render_mode == "ansi":
                print(env.render())
            elif env.render_mode == "human":
                time.sleep(delay_sec)
    except Exception:
        return


def run() -> None:
    parser = argparse.ArgumentParser(description="FrozenLake linear-algebra policy evaluation (gymnasium-based)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--p", type=float, default=0.8, help="Probability of frozen tiles when generating a random map.")
    parser.add_argument(
        "--render",
        choices=["none", "ansi", "human"],
        default="ansi",
        help="If 'human' fails, it falls back to 'ansi'.",
    )
    args = parser.parse_args()

    render_mode = None if args.render == "none" else args.render

    spec = FrozenLakeSpec(gamma=0.9)
    try:
        env = make_env(size=args.size, p=args.p, seed=args.seed, render_mode=render_mode)
    except Exception as e:
        if render_mode == "human":
            print(f"[warn] Failed to create human renderer ({e!r}). Falling back to ansi.")
            render_mode = "ansi"
            env = make_env(size=args.size, p=args.p, seed=args.seed, render_mode=render_mode)
        else:
            raise
    try:
        env.reset(seed=args.seed)

        P_pi, R_pi = build_random_policy_matrices(env)
        V = solve_value_function(P_pi, R_pi, gamma=spec.gamma)

        nrow = env.unwrapped.nrow  # type: ignore[attr-defined]
        ncol = env.unwrapped.ncol  # type: ignore[attr-defined]
        print(f"State-value function V ({nrow}x{ncol}):")
        pretty_print_value(V, env)

        path = greedy_path_from_value(env, V, start=0, goal=None)
        print("Greedy path:")
        print(" -> ".join(map(str, path)))

        render_greedy_rollout(env, path)
    finally:
        env.close()


if __name__ == "__main__":
    run()


#########################################################
# TODO: 아래 질문에 답해주세요.
#
# Q: (Part 2에서처럼) 선형대수로 연립방정식을 풀어 Bellman 기대 방정식의 해를 구할 수 있습니다.
#    그렇다면 최적(Optimality) 벨만 방정식도 같은 방식으로 선형 연립방정식으로 풀 수 있을까요?
#    가능/불가능하다면 그 이유는 무엇인가요?
# A: 
#
#########################################################