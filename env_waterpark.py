import numpy as np
import random

def get_influx_multiplier(hour):
    if 14 <= hour < 17:
        return 1.0
    elif 9 <= hour < 12:
        return 0.7
    elif 12 <= hour < 14 or 17 <= hour < 19:
        return 0.3
    else:
        return 0.0

class WaterParkEnv:
    def __init__(self, max_steps=60, max_replace=20):
        self.max_steps = max_steps
        self.max_replace = max_replace
        self.reset()

    def get_current_guests(self, step):
        hour = 9 + (step * 10) // 60
        if 9 <= hour < 12:
            return int(10000 * 0.3 / 18)
        elif 12 <= hour < 14:
            return int(10000 * 0.1 / 12)
        elif 14 <= hour < 17:
            return int(10000 * 0.5 / 18)
        else:
            return int(10000 * 0.1 / 12)

    def is_all_optimal(self, state):
        ammonia, turbidity, ph, *_ = state
        return (ammonia <= 0.5) and (turbidity <= 2.8) and (5.8 <= ph <= 8.6)

    def is_all_over(self, state):
        ammonia, turbidity, ph, *_ = state
        return (ammonia > 0.5) and (turbidity > 2.8) and (ph < 5.8 or ph > 8.6)

    def reset(self):
        self.state = np.array([
            random.uniform(0, 0.5),
            random.uniform(0, 2.8),
            random.uniform(5.8, 8.6),
            self.max_replace,
            0
        ])
        self.steps = 0
        self.replace_count = 0
        self.done = False
        return self.state.copy()

    def step(self, action):
        ammonia, turbidity, ph, replace_left, current_step = self.state
        reward = 0
        done = False
        hour = 9 + (int(current_step) * 10) // 60
        influx_multiplier = get_influx_multiplier(hour)
        if action == 1 and replace_left > 0:
            self.state = np.array([
                random.uniform(0, 0.2),
                random.uniform(0, 2.0),
                random.uniform(6.0, 8.0),
                replace_left - 1,
                current_step + 1
            ])
            self.replace_count += 1
        elif action == 1 and replace_left <= 0:
            self.state[4] += 1
        else:
            #ph +-3.5만큼 변동
            ph_change = random.uniform(-3.0, 3.0) * influx_multiplier
            new_ph = ph + ph_change
            #탁도 최대 +5만큼 변동
            new_turbidity = turbidity + random.uniform(3.0, 5.0) * influx_multiplier
            #암모니아 최대 +7만큼 변동
            new_ammonia = ammonia + random.uniform(3.0, 7.0) * influx_multiplier
            self.state = np.array([
                new_ammonia,
                new_turbidity,
                new_ph,
                replace_left,
                current_step + 1
            ])
        self.steps += 1

        # 기준 초과 개수
        exceed_count = 0
        if self.state[0] > 0.5:
            exceed_count += 1
        if self.state[1] > 2.8:
            exceed_count += 1
        if self.state[2] < 5.8 or self.state[2] > 8.6:
            exceed_count += 1

        #보상 함수: 기준 초과 개수 및 액션
        if self.is_all_optimal(self.state) and action == 1:
            reward = -1.0  #최적 상태에서 교체(자원 낭비)
        elif exceed_count == 1: #1개 초과
            if action == 1: #교체
                reward = 0.3
            else: #유지
                reward = -0.1
        elif exceed_count == 2: #2개 초과
            if action == 1: #교체
                reward = 0.6
            else: #유지
                reward = -0.7
        elif exceed_count == 3: #3개 모두 초과
            if action == 1: #교체
                reward = 0.8
            else: #유지
                reward = -1.0
        elif self.is_all_optimal(self.state) and action == 0: #모두 정상
            reward = 0.6 #유지
            
        # #모든 수질이 최적인데 물교체(큰 패널티)
        # if self.is_all_optimal(self.state) and action == 1:
        #     reward = -0.5
        # #모든 수질이 기준을 넘었는데 물교체x(큰 패널티)
        # elif self.is_all_over(self.state) and action == 0:
        #     reward = -0.5
        # #하나 이상 기준 넘었는데 물 교체(약간의 자원 낭비, 수질 정화)
        # elif (self.state[0] > 0.5 or self.state[1] > 2.8 or self.state[2] < 5.8 or self.state[2] > 8.6) and action == 1:
        #     reward = 0.3
        # #하나 이상 기준 넘었는데 물 교체 안함(자원 절약, 수질 악화)
        # elif (self.state[0] > 0.5 or self.state[1] > 2.8 or self.state[2] < 5.8 or self.state[2] > 8.6) and action == 0:
        #     reward = -0.3
        # #모든 수질이 최적인데 유지(기본 보상)
        # elif (self.state[0] <= 0.5 and self.state[1] <= 2.8 and 5.8 <= self.state[2] <= 8.6) and action == 0:
        #     reward = 0.7

        #교체할 물이 없는데 교체 시도
        if action == 1 and replace_left <= 0:
            reward -= 0.4

        # 기존 reward 산정 이후 아래 패널티 추가
        replace_penalty = 0
        if action == 1:  # 물 교체 시도 시
            replace_penalty = -0.4 # 교체 한 번 당 페널티
        reward += replace_penalty


        if self.state[4] >= self.max_steps or self.steps >= self.max_steps:
            done = True
        self.done = done
        return self.state.copy(), reward, done, {
            'ammonia': self.state[0],
            'turbidity': self.state[1],
            'ph': self.state[2],
            'replace_left': self.state[3],
            'step': self.state[4],
            'guests': self.get_current_guests(int(self.state[4]))
        }
