import numpy as np
import matplotlib.pyplot as plt

from env_waterpark_fixed import WaterParkEnv
from agent_waterpark import QAgent, FixedIntervalPolicy, quantize_state

# 그리디 정책 클래스 (학습된 Q-table에서 가장 큰 값 선택)
class GreedyPolicy:
    def __init__(self, q_table):
        self.q_table = q_table

    def choose_action(self, state):
        return np.argmax(self.q_table[state])

# 실험에 사용하는 환경 목록
CSV_LIST = [
    "fixed_env_changes1.csv",
    "fixed_env_changes2.csv",
    "fixed_env_changes3.csv",
    "fixed_env_changes4.csv"
]

# 실험 환경별 라벨 (그래프 제목용)
ENV_LABELS = [
    "Env 1 (80% influx)",
    "Env 2 (100%)",
    "Env 3 (120%)",
    "Env 4 (140%)"
]

# 주어진 정책을 그대로 실행하여 성능 계산
def run_policy(env, policy, quantize, episodes):
    total_rewards, replace_counts = [], []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            s = quantize_state(state) if quantize else state
            action = policy.choose_action(s)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        # 남은 교체 횟수 보너스 추가
        total_reward += 0.1 * state[3]
        total_rewards.append(total_reward)
        replace_counts.append(env.replace_count)

    return total_rewards, replace_counts

# Q-Learning 기반 학습
def train_qlearning(env, agent, episodes):
    rewards, replaces = [], []

    for ep in range(episodes):
        state = env.reset()
        state_disc = quantize_state(state)
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state_disc)
            next_state, reward, done, _ = env.step(action)
            next_state_disc = quantize_state(next_state)

            # Q-table 업데이트
            agent.learn(state_disc, action, reward, next_state_disc)

            total_reward += reward
            state_disc = next_state_disc
            state = next_state

        total_reward += 0.1 * state[3]
        rewards.append(total_reward)
        replaces.append(env.replace_count)
        agent.decay_epsilon()

    return rewards, replaces

# 이동 평균 계산 (그래프 smooth 처리용)
def moving_average(data, window=30):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# 메인 실행
if __name__ == "__main__":
    episodes = 2000

    # 환경 수 만큼 subplot 구성 (4개 환경 x 2열)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 16))
    fig.suptitle("Q-Learning vs Fixed vs Greedy - Across 4 Fixed Environments", fontsize=16)

    for idx, csv_file in enumerate(CSV_LIST):
        print(f"▶ 실험 환경: {csv_file}")

        env = WaterParkEnv(csv_file)
        q_agent = QAgent(epsilon=0.1, epsilon_decay=0.9995, epsilon_min=0.01)
        fixed_policy = FixedIntervalPolicy()
        greedy_policy = GreedyPolicy(q_agent.Q_table)

        # 학습 및 정책 평가
        q_rewards, q_replaces = train_qlearning(env, q_agent, episodes)
        greedy_rewards, greedy_replaces = run_policy(env, greedy_policy, quantize=True, episodes=episodes)
        fixed_rewards, fixed_replaces = run_policy(env, fixed_policy, quantize=False, episodes=episodes)

        row = idx

        # ▶ 리워드 그래프 (왼쪽)
        ax1 = axes[row][0]
        ax1.set_title(f"{ENV_LABELS[idx]} - Reward")
        ax1.plot(moving_average(fixed_rewards), label="Fixed")
        ax1.plot(moving_average(q_rewards), label="Q-Learning")
        ax1.plot(moving_average(greedy_rewards), label="Greedy")
        if row == 0:
            ax1.legend()
        ax1.set_ylabel("Reward")

        # ▶ 교체 횟수 그래프 (오른쪽)
        ax2 = axes[row][1]
        ax2.set_title(f"{ENV_LABELS[idx]} - Replacements")
        ax2.plot(moving_average(fixed_replaces), label="Fixed")
        ax2.plot(moving_average(q_replaces), label="Q-Learning")
        ax2.plot(moving_average(greedy_replaces), label="Greedy")
        if row == 0:
            ax2.legend()
        ax2.set_ylabel("Water Replacements")

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 타이틀 영역 확보
    plt.show()
