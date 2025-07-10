# Reinforcement Learning-Based Water Purification System

## Overview
워터파크의 물 교체 시점을 강화학습(Q-Learning)과 규칙기반 정책으로 비교하는 시뮬레이션 프로젝트입니다.

## Features
- 워터파크 환경 시뮬레이션
- Q-러닝/고정주기 정책 비교
- 자원 절약 & 수질 유지 보상 구조
- 성능 변화 그래프 시각화

## Structure
- env_waterpark.py: 환경 정의
- agent_waterpark.py: 에이전트/정책
- train_waterpark.py: 학습/실험/시각화

## Usage
1. 설치: `pip install numpy matplotlib`
2. 실행: `python train_waterpark.py`