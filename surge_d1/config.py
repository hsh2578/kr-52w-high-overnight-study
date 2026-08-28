# -*- coding: utf-8 -*-
"""프로젝트 전역 상수와 경로. 스펙 §3.1, §6, §7.2 참조."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- 이벤트 정의 (스펙 §3.1) ---
HIGH_52W_PERIOD = 252        # 52주 = 252거래일
VOLUME_SURGE_MULT = 1.5      # 당일 거래량 / 50일 평균
MIN_TRADING_VALUE = 1e10     # 거래대금 100억 (원)
MIN_BAR_INDEX = 260          # 피처 계산에 220봉 필요 + 여유
HALT_LOOKBACK = 5            # 이벤트 직전 이 봉수 안에 거래정지가 있으면 제외

# --- 라벨 (스펙 §6) ---
COST = 0.0021                # 왕복 거래비용

# --- 검증 (스펙 §7.2) ---
EMBARGO_DAYS = 5
RANDOM_SEED = 42

# --- 경로 ---
HOLLYKR_CACHE = Path(
    r"C:\Users\hsh\Desktop\vibecoding\주식 자동매매시스템과 관련  프로젝트\.cache"
)
CACHE_PATH = PROJECT_ROOT / "data" / "ohlcv_cache.pkl"
CATALYST_CSV = Path(
    r"C:\Users\hsh\Desktop\모의투자대회\data\특징주_재료_상승률_2년.csv"
)
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.parquet"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs"
