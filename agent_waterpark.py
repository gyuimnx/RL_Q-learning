import numpy as np
import random

# 상태 양자화 함수 (수질, 교체 남은 횟수, 시간대)
def quantize_state(state):
    ammonia, turbidity, ph, replace_left, current_step = state

    # pH: 0=low, 1=optimal, 2=high
    if ph < 5.8:
        ph_state = 0
    elif ph > 8.6:
        ph_state = 2
    else:
        ph_state = 1

    # 탁도: 0=optimal, 1=high
    if turbidity <= 2.8:
        turbidity_state = 0
    else:
        turbidity_state = 1

    # 암모니아: 0=optimal, 1=high
    if ammonia <= 2.8:
        ammonia_state = 0
    else:
        ammonia_state = 1

    # 남은 교체 횟수: 0=없음, 1=1~5, 2=6~10, 3=11~15, 4=16~20(다섯 단계로 범주화)
    if replace_left == 0:
        replace_state = 0
    elif replace_left <= 5:
        replace_state = 1
    elif replace_left <= 10:
        replace_state = 2
    elif replace_left <= 15:
        replace_state = 3
    else:
        replace_state = 4

    # 시간대: 0=9~12시(아침), 1=12~14시(점심), 2=14~17시(오후), 3=17~19시(저녁)
    hour = 9 + (int(current_step) * 10) // 60
    if 9 <= hour < 12:
        time_state = 0  # 아침
    elif 12 <= hour < 14:
        time_state = 1  # 점심
    elif 14 <= hour < 17:
        time_state = 2  # 오후
    else:
        time_state = 3  # 저녁

    # 양자화된 상태를 튜플로 반환
    return (ammonia_state, turbidity_state, ph_state, replace_state, time_state)

# Q-러닝 에이전트
class QAgent:
    # shape: 암모니아 2단계, 탁도 2단계 pH 3단계, 남은 교체 횟수 5단계, 시간대 4단계
    # n_actions : 액션 개수(0 유지, 1 교체)
    # alpha : 학습률(새로 얻은 보상을 Q-table에 얼마나 반영할지-크면 많이 작으면 적게)
    # gamma : 할인율(미래 보상을 얼마나 중요하게 여길지-1에 가까울수록 미래 보상까지 최대화)
    # epsilon : 탐험률(랜덤)
    # epsilon_min : 탐험률 최소값(후반에 최소 랜덤)
    # epsilon_decay : 감쇠 비율(탐험률 얼마나 줄일지)
    def __init__(self, state_shape=(2,2,3,5,4), n_actions=2, alpha=0.1, gamma=0.95, epsilon=0.1, epsilon_decay=0.0):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.Q_table = np.zeros(state_shape + (n_actions,))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon

    # e-greedy 정책
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q_table[state])

    # 벨만 방적식 기반 Q-value 업데이트
    def learn(self, state, action, reward, next_state):
        best_next = np.max(self.Q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q_table[state + (action,)]
        self.Q_table[state + (action,)] += self.alpha * td_error

    # 학습이 진행될수록 탐험률이 점진적으로 감소
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 고정주기(규칙기반) 정책 - 30분마다(3스텝마다) 남은 교체 횟수가 있으면 교체, 아니면 유지
class FixedIntervalPolicy:
    def choose_action(self, state):
        _, _, _, replace_left, current_step = state
        if replace_left > 0 and int(current_step) % 3 == 0:
            return 1
        return 0

# 랜덤 정책(비교용)
class RandomPolicy:
    def choose_action(self, state):
        return random.choice([0, 1])
