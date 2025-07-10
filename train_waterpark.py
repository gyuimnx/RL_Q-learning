import numpy as np
import matplotlib.pyplot as plt
from env_waterpark import WaterParkEnv
from agent_waterpark import QAgent, FixedIntervalPolicy, quantize_state

# 주어진 정책으로 여러 에피소드 동안 환경 실행 후 각 에피소드의 총 보상을 리스트로 저장
def run_policy(env, policy, quantize=False, episodes=5000):
    all_rewards = []
    for ep in range(episodes):
        state = env.reset()
        rewards = 0
        done = False
        while not done:
            s = quantize_state(state) if quantize else state
            action = policy.choose_action(s)
            state, reward, done, info = env.step(action)
            rewards += reward
        all_rewards.append(rewards)
    return all_rewards

# Q-Learning 학습 함수
# Q-Learning 에이전트가 환경에서 1000회 학습
def train_qlearning(env, agent, episodes=5000):
    rewards_q = []
    for ep in range(episodes):
        state = env.reset()
        state_disc = quantize_state(state)
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state_disc)
            next_state, reward, done, info = env.step(action)
            next_state_disc = quantize_state(next_state)
            agent.learn(state_disc, action, reward, next_state_disc)
            state_disc = next_state_disc
            total_reward += reward
        rewards_q.append(total_reward)
        agent.decay_epsilon()
        if (ep+1) % 100 == 0: # 100회마다 진행상황(에피소드 번호, 탐험률) 출력
            print(f"Episode {ep+1} completed. Epsilon: {agent.epsilon:.3f}")
    return rewards_q

# 환경, 큐러닝 에이전트, 고정주기 정책 객체 생성
if __name__ == "__main__":
    env = WaterParkEnv()
    q_agent = QAgent()
    fixed_policy = FixedIntervalPolicy()

    # 기존 정책(30분마다 교체) 에피소드별 보상 저장
    rewards_fixed = run_policy(env, fixed_policy, quantize=False, episodes=5000)
    # Q-러닝 학습 및 에피소드별 보상 저장
    rewards_q = train_qlearning(env, q_agent, episodes=5000)

    # Q-러닝 학습 후, 탐욕 정책으로 평가(20회)
    class GreedyQPolicy:
        def choose_action(self, state):
            return np.argmax(q_agent.Q_table[state])
    rewards_q_eval = run_policy(env, GreedyQPolicy(), quantize=True, episodes=50)
    mean_fixed = np.mean(rewards_fixed[-50:])
    mean_q = np.mean(rewards_q_eval)

    # 이동평균 함수
    def moving_average(data, window_size=50):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # Q-러닝 vs 기존 정책: 에피소드별 평균 보상 변화(이동평균)만 선 그래프로 시각화
    plt.figure(figsize=(10,6))
    plt.plot(moving_average(rewards_fixed, 50), label='Fixed(30min)', color='skyblue')
    plt.plot(moving_average(rewards_q, 50), label='Q-Learning', color='limegreen')
    plt.xlabel('Episode')
    plt.ylabel('Mean Total Reward (Moving Average)')
    plt.title('Policy Performance Comparison (Moving Average)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"최근 20회 기준 30분마다 교체 정책 평균 보상: {mean_fixed:.2f}")
    print(f"Q-러닝 정책(탐욕) 평균 보상: {mean_q:.2f}")
