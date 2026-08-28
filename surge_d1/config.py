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

# 코스피/코스닥 일일 가격제한폭은 ±30%이나, 상한가는 호가단위(틱사이즈)로
# 반올림돼 실제로는 30%를 살짝 넘는 값으로 찍힌다(실측: 33건 중 32건이
# [30.001%, 30.148%] 구간). 순수 30%를 기준으로 배제하면 정상적인 상한가
# 이벤트까지 잘려나가므로, 0.32로 여유를 두어 정상 상한가는 모두 살리고
# 물리적으로 불가능한 값(예: 32%+)만 데이터 오류로 보고 배제한다.
MAX_DAILY_MOVE = 0.32

# 상한가 판정 기준. MAX_DAILY_MOVE와 목적이 다르다 — 저쪽은 '물리적으로
# 불가능한 손상 데이터'를 버리는 것이고, 이쪽은 '정상적이지만 종가에 살 수
# 없는 봉'을 식별하는 것이다.
#
# 29.5%로 두는 근거: 돌파일 등락률 분포에 뚜렷한 공백이 있다 — 29.0~29.5%
# 구간은 41건뿐인데 29.5~29.9%가 755건, 29.9~30.1%가 1,217건이다. 호가단위
# 반올림 때문에 상한가가 정확히 30%가 아니라 그 아래에 흩어져 찍히므로,
# 공백 아래쪽인 29.5%를 경계로 삼으면 상한가 무리를 통째로 잡는다.
#
# 왜 중요한가: 상한가는 종가 동시호가에서 매수 잔량만 쌓여 체결이 극히
# 불확실한데, 실측상 모델 픽의 74%가 여기에 몰려 있었다(리포트 §5-5).
LIMIT_UP_PCT = 0.295

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
