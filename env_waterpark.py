import numpy as np
import random

class WaterParkEnv:
    """
    워터파크 물 교체 환경 클래스
    상태: [암모니아, 탁도, pH, 남은 교체 횟수, 현재 스텝]
    액션: 0(유지), 1(물 교체)
    """
    def __init__(self, max_steps=60, max_replace=20): # max_replace : 하루 최대 교체 횟수(자원 제한)
        self.max_steps = max_steps  # 9시~19시, 10분 단위 60스텝
        self.max_replace = max_replace
        self.reset()

    # 시간대별 이용객 수
    def get_current_guests(self, step):
        hour = 9 + (step * 10) // 60 # step을 실제 시간(시)으로 변환
        if 9 <= hour < 12:
            # 9~12시: 18스텝, 3,000명
            return int(10000 * 0.3 / 18)
        elif 12 <= hour < 14:
            # 12~14시: 12스텝, 1,000명
            return int(10000 * 0.1 / 12)
        elif 14 <= hour < 17:
            # 14~17시: 18스텝, 5,000명
            return int(10000 * 0.5 / 18)
        else:
            # 17~19시: 12스텝, 1,000명
            return int(10000 * 0.1 / 12)

    # 상태 평가
    def is_all_optimal(self, state): # 모든 수질 지표가 최적인지
        ammonia, turbidity, ph, *_ = state
        return (ammonia <= 0.5) and (turbidity <= 2.8) and (5.8 <= ph <= 8.6)

    def is_all_over(self, state): # 모든 수질 지표가 기준을 초과했는지
        ammonia, turbidity, ph, *_ = state
        return (ammonia > 0.5) and (turbidity > 2.8) and (ph < 5.8 or ph > 8.6)

    # 환경 상태 초기화
    def reset(self):
        self.state = np.array([
            random.uniform(0, 0.5),    # 암모니아(나중에 결합잔류염소로 교체 가능)
            random.uniform(0, 2.8),    # 탁도
            random.uniform(5.8, 8.6),  # pH
            self.max_replace,          # 남은 교체 횟수
            0                          # 현재 스텝(0=9시)
        ])
        self.steps = 0
        self.replace_count = 0
        self.done = False
        return self.state.copy()

    # 환경 동작(현재 이용객 수에 따라 오염 증가량 계산)
    def step(self, action):
        ammonia, turbidity, ph, replace_left, current_step = self.state
        reward = 0
        done = False

        guests = self.get_current_guests(int(current_step))
        ammonia_increase = guests * 0.00001 + random.uniform(0, 0.01)
        turbidity_increase = guests * 0.00002 + random.uniform(0, 0.02)

        # 상태 변화
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
            new_ammonia = ammonia + ammonia_increase
            new_turbidity = turbidity + turbidity_increase
            new_ph = ph + random.uniform(-0.03, 0.03)
            self.state = np.array([
                new_ammonia,
                new_turbidity,
                new_ph,
                replace_left,
                current_step + 1
            ])

        self.steps += 1

        # -------------- 보상 구조 --------------
        # 1. 모두 최적 상태에서 물 교체
        if self.is_all_optimal(self.state) and action == 1:
            reward = -1.0
        # 2. 모두 기준 초과인데 물 교체 안함
        elif self.is_all_over(self.state) and action == 0:
            reward = -1.0
        # 3. 기준 초과 상태에서 물 교체(하나라도 넘었을 때)
        elif (self.state[0] > 0.5 or self.state[1] > 2.8 or self.state[2] < 5.8 or self.state[2] > 8.6) and action == 1:
            reward = 0.3
        # 4. 기준 초과 상태에서 물 교체 안함(하나라도 넘었을 때)
        elif (self.state[0] > 0.5 or self.state[1] > 2.8 or self.state[2] < 5.8 or self.state[2] > 8.6) and action == 0:
            reward = -0.2
        # 5. 기준 내에서 물 교체 안함
        elif (self.state[0] <= 0.5 and self.state[1] <= 2.8 and 5.8 <= self.state[2] <= 8.6) and action == 0:
            reward = 0.5
        # # 6. 기준 내에서 물 교체함
        # elif (self.state[0] <= 0.5 and self.state[1] <= 2.8 and 5.8 <= self.state[2] <= 8.6) and action == 1:
        #     reward = -1.0

        # 교체 한도 초과 시 추가 패널티
        if action == 1 and replace_left <= 0:
            reward -= 0.7

        # # 수질 기준 초과 시 에피소드 종료(종료 조건)
        # if (
        #     self.state[0] > 0.5 or
        #     self.state[1] > 2.8 or
        #     self.state[2] < 5.8 or self.state[2] > 8.6
        # ):
        #     done = True

        # 영업 종료(19시) 또는 최대 스텝 도달 시 종료
        if self.state[4] >= self.max_steps or self.steps >= self.max_steps:
            done = True

        self.done = done
        return self.state.copy(), reward, done, {
            'ammonia': self.state[0],
            'turbidity': self.state[1],
            'ph': self.state[2],
            'replace_left': self.state[3],
            'step': self.state[4],
            'guests': guests
        }