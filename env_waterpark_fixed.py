import numpy as np
import pandas as pd

class WaterParkEnv:
    """
    고정 환경(Randomness 없음). fixed_env_changes*.csv의 수질 변화량 데이터를 사용.
    """

    def __init__(self, csv_path, max_steps=60, max_replace=20):
        """
        Args:
            csv_path (str): pH, turbidity, ammonia 변화량이 들어있는 CSV 파일
            max_steps (int): 환경 스텝 수 (기본: 60분)
            max_replace (int): 물 교체 최대 횟수
        """
        self.csv = pd.read_csv(csv_path)
        self.max_steps = max_steps
        self.max_replace = max_replace
        self.reset()

    def reset(self):
        """
        상태 초기화 (결정론적 초기값)

        Returns:
            ndarray: 초기 상태 [ammonia, turbidity, pH, replace_left, timestep]
        """
        self.state = np.array([
            0.2,   # ammonia
            1.5,   # turbidity
            7.2,   # pH
            self.max_replace,
            0
        ])
        self.steps = 0
        self.replace_count = 0
        self.done = False
        return self.state.copy()

    def get_current_guests(self, step):
        """환경 구조 통일용 인터페이스 (의미 X)"""
        return 100  # placeholder for compatibility

    def is_all_optimal(self, state):
        ammonia, turbidity, ph, *_ = state
        return (ammonia <= 0.5) and (turbidity <= 2.8) and (5.8 <= ph <= 8.6)

    def is_all_over(self, state):
        ammonia, turbidity, ph, *_ = state
        return (ammonia > 0.5) and (turbidity > 2.8) and (ph < 5.8 or ph > 8.6)

    def step(self, action):
        """
        한 스텝 진행 (동일한 시나리오로 수질 변화)

        Args:
            action (int): 0 = 유지 / 1 = 교체

        Returns:
            next_state, reward, done, info
        """
        ammonia, turbidity, ph, replace_left, current_step = self.state

        row = self.csv.iloc[min(int(current_step), len(self.csv) - 1)]
        ph_delta = row['pH_change']
        turbidity_delta = row['turbidity_change']
        ammonia_delta = row['ammonia_change']

        if action == 1 and replace_left > 0:
            # 물을 교체하면 수질 초기화
            self.state = np.array([0.1, 1.0, 7.2, replace_left - 1, current_step + 1])
            self.replace_count += 1
        elif action == 1 and replace_left <= 0:
            # 물 없음에도 교체 시도
            self.state[4] += 1
        else:
            # 수질 변화 적용
            ph = ph + ph_delta
            turbidity = turbidity + turbidity_delta
            ammonia = ammonia + ammonia_delta
            self.state = np.array([ammonia, turbidity, ph, replace_left, current_step + 1])
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
                reward = 0.2
            else: #유지
                reward = -0.1
        elif exceed_count == 2: #2개 초과
            if action == 1: #교체
                reward = 0.5
            else: #유지
                reward = -0.6
        elif exceed_count == 3: #3개 모두 초과
            if action == 1: #교체
                reward = 0.8
            else: #유지
                reward = -1.0
        elif self.is_all_optimal(self.state) and action == 0: #모두 정상
            reward = 0.6 #유지
            
        #교체할 물이 없는데 교체 시도
        if action == 1 and replace_left <= 0:
            reward -= 0.4

        # 기존 reward 산정 이후 아래 패널티 추가
        replace_penalty = 0
        if action == 1:  # 물 교체 시도 시
            replace_penalty = -0.36 # 교체 한 번 당 페널티
        reward += replace_penalty

        done = False
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
