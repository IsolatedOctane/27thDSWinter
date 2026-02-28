import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# ==================================================
# 1. Actor-Critic 네트워크 정의
# ==================================================
class Actor(nn.Module):
    """정책 네트워크: 상태 → 행동 logits"""  
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # logits 출력 (Categorical에서 softmax 처리)


class Critic(nn.Module):
    """가치 네트워크: 상태 → V(s)"""
    def __init__(self, state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ==================================================
# 2. Trajectory Buffer (에피소드 데이터 저장)
# ==================================================
class TrajectoryBuffer:
    """PPO는 여러 에피소드의 경험을 모아서 배치 업데이트합니다."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def add(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def get(self):
        return (torch.tensor(np.array(self.states), dtype=torch.float32),
                torch.tensor(self.actions, dtype=torch.long),
                torch.tensor(self.rewards, dtype=torch.float32),
                torch.tensor(np.array(self.next_states), dtype=torch.float32),
                torch.tensor(self.dones, dtype=torch.float32),
                torch.tensor(self.log_probs, dtype=torch.float32))

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []


# ==================================================
# 3. PPO 에이전트
# ==================================================
class PPOAgent:
    def __init__(self):
        self.gamma = 0.99
        self.gae_lambda = 0.95          # GAE의 lambda (bias-variance 트레이드오프)
        self.lr_actor = 0.0003
        self.lr_critic = 0.001
        self.clip_range = 0.2           # PPO 클리핑 범위 (epsilon)
        self.n_epochs = 10              # 같은 데이터로 반복 학습 횟수

        self.state_size = 4   # CartPole 관측 차원
        self.action_size = 2  # CartPole 행동 수

        self.actor = Actor(self.state_size, self.action_size)
        self.critic = Critic(self.state_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.buffer = TrajectoryBuffer()

    @torch.no_grad()
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    @torch.no_grad()
    def compute_gae(self, states, rewards, next_states, dones):
        """GAE (Generalized Advantage Estimation) 계산

        GAE는 TD error의 지수가중 합으로 advantage를 추정합니다:
          A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...
          delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)  (TD error)
        """
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze() * (1 - dones)

        T = len(rewards)
        advantages = torch.zeros(T)

        # TODO 1: GAE를 계산하세요.
        # 역순으로 순회하면서 delta(TD error)와 gae를 누적 계산합니다.
        #
        # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
        # gae = delta + gamma * lambda * (1 - done) * gae
        # advantages[t] = gae
        #
        # 마지막으로 returns = advantages + values 를 계산합니다.
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] - values[t] # TODO
            gae = delta + self.gamma * self.gae_lambda * (1- dones[t]) * gae    # TODO
            advantages[t] = gae

        returns = advantages + values # TODO

        return advantages, returns

    def update(self):
        """PPO 업데이트: 수집된 trajectory로 여러 에포크 학습"""
        states, actions, rewards, next_states, dones, old_log_probs = self.buffer.get()

        advantages, returns = self.compute_gae(states, rewards, next_states, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            # === Actor 업데이트 ===
            logits = self.actor(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)

            # TODO 2: PPO Clipped Surrogate Loss를 계산하세요.
            # 1) ratio = exp(new_log_probs - old_log_probs)
            # 2) surr1 = ratio * advantages
            # 3) surr2 = clamp(ratio, 1-eps, 1+eps) * advantages
            # 4) actor_loss = -min(surr1, surr2).mean()
            ratio = torch.exp(new_log_probs- old_log_probs)      # TODO
            surr1 = ratio * advantages       # TODO
            surr2 = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * advantages      # TODO
            actor_loss = -torch.min(surr1, surr2).mean()  # TODO

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # === Critic 업데이트 ===
            values_pred = self.critic(states).squeeze()

            # TODO 3: Critic loss (MSE)를 계산하세요.
            # 힌트: nn.functional.mse_loss(values_pred, returns)
            critic_loss = nn.functional.mse_loss(values_pred, returns)  # TODO

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        self.buffer.clear()


# ==================================================
# 4. 학습 루프
# ==================================================
episodes = 1000
update_interval = 5

env = gym.make('CartPole-v1')
agent = PPOAgent()
reward_history = []

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.buffer.add(state, action, reward, next_state, done, log_prob)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)

    if (episode + 1) % update_interval == 0:
        agent.update()

    if episode % 100 == 0:
        print(f"episode: {episode}, total reward: {total_reward:.1f}")


# ==================================================
# 5. 결과 시각화
# ==================================================
target_reward = 400
plt.plot(reward_history)
plt.title("PPO (Proximal Policy Optimization)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.axhline(y=target_reward, color='r', linestyle='--', label='Target')
plt.legend()
plt.savefig("ppo.png", dpi=150, bbox_inches="tight")
plt.show()


# ==================================================
# 6. 학습된 에이전트 시연 (GUI)
# ==================================================
env.close()
demo_env = gym.make('CartPole-v1', render_mode='human')

for i in range(3):
    state, info = demo_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = agent.get_action(state)
        state, reward, terminated, truncated, info = demo_env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Demo {i+1}: total reward = {total_reward:.0f}")

demo_env.close()


# [과제 답변란]
'''
질문: PPO의 Clipping이 왜 필요한지 설명하고, A2C(Week6) 대비 어떤 점이 개선되었는지 서술하시오.
답변: 과도한 정책 업데이트를 방지하고, 1-step TD 방식인 A2C보다 여러 step을 거쳐 안정적인 학습이 가능하다. 또한, sample 효율성이 좋다.

'''
