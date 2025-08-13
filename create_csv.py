import csv

# 시간대별 정보
influx_per_time = [166, 83, 277, 83]
max_ph_change, max_turbidity_change, max_ammonia_change = 3.5, 3.0, 1.0
max_influx = max(influx_per_time)

# 시간대별 변화값 산정
ph_changes = [round(max_ph_change * (i / max_influx), 2) for i in influx_per_time]
turbidity_changes = [round(max_turbidity_change * (i / max_influx), 2) for i in influx_per_time]
ammonia_changes = [round(max_ammonia_change * (i / max_influx), 2) for i in influx_per_time]
steps_per_time = [18, 12, 18, 12]  # 각 시간대별 스텝 수

ph_step_changes, turbidity_step_changes, ammonia_step_changes = [], [], []
for idx, cnt in enumerate(steps_per_time):
    ph_step_changes.extend([ph_changes[idx]] * cnt)
    turbidity_step_changes.extend([turbidity_changes[idx]] * cnt)
    ammonia_step_changes.extend([ammonia_changes[idx]] * cnt)

# CSV 파일 저장
with open('fixed_env_changes2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'pH_change', 'turbidity_change', 'ammonia_change'])
    for step in range(60):
        writer.writerow([step + 1, ph_step_changes[step], turbidity_step_changes[step], ammonia_step_changes[step]])