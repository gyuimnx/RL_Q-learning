import numpy as np
import random

def quantize_state(state):
    ammonia, turbidity, ph, replace_left, current_step = state
    if ph < 5.8: #pH 낮음
        ph_state = 0
    elif ph > 8.6: #pH 높음
        ph_state = 2
    else: #pH 정상
        ph_state = 1
        
    #탁도 2.8 이하면 0
    turbidity_state = 0 if turbidity <= 2.8 else 1
    #이거 잔류염소로 바꿀 예정
    ammonia_state = 0 if ammonia <= 2.8 else 1
    
    #이건 뭐지(물 교체 횟수)
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
        
    #시간 양자화 -> 아침 오후 저녁으로 다시 해야함
    hour = 9 + (int(current_step) * 10) // 60
    if 9 <= hour < 12:
        time_state = 0
    elif 12 <= hour < 14:
        time_state = 1
    elif 14 <= hour < 17:
        time_state = 2
    else:
        time_state = 3
        
    return (ammonia_state, turbidity_state, ph_state, replace_state, time_state)

class QAgent:
    def __init__(self, state_shape=(2,2,3,5,4), n_actions=2, alpha=0.1, gamma=0.95, epsilon=0.1, epsilon_decay=0.0, epsilon_min=0.01):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.Q_table = np.zeros(state_shape + (n_actions,))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min  #나중에 빼도 됨

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q_table[state])

    def learn(self, state, action, reward, next_state):
        best_next = np.max(self.Q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q_table[state + (action,)]
        self.Q_table[state + (action,)] += self.alpha * td_error

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

class FixedIntervalPolicy:
    def choose_action(self, state):
        _, _, _, replace_left, current_step = state
        if replace_left > 0 and int(current_step) % 3 == 0:
            return 1
        return 0

class RandomPolicy:
    def choose_action(self, state):
        return random.choice([0, 1])
