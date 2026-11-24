# -*- coding: utf-8 -*-
"""프로젝트 전역 상수와 경로. 스펙 §3.1, §6, §7.2 참조."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 이 저장소에 들어 있지 않은 자료의 위치. PC마다 달라서 local_paths.json(커밋하지 않음)에
# 적는다. 그 파일이 없으면 data/external/ 아래의 같은 이름을 본다.
# 키 목록은 local_paths.example.json 참고.
_LOCAL_FILE = PROJECT_ROOT / "local_paths.json"
_LOCAL = json.loads(_LOCAL_FILE.read_text(encoding="utf-8")) if _LOCAL_FILE.exists() else {}


def external(key: str) -> Path:
    return Path(_LOCAL.get(key, PROJECT_ROOT / "data" / "external" / key))

# --- 이벤트 정의 ---
# 스크리너 실전 조건을 그대로 옮겼다. 종가 252봉 신고가라는 단순 정의 대신
# 이 조건을 쓰는 이유: 실측상 이벤트는 23,654 -> 6,443건으로 줄지만 평균
# 수익이 +0.4747% -> +0.5956%로 **올라간다**. 150일선 추세 필터와 갭 체크가
# "이미 추세에 있고, 방금 처음 넘은" 돌파만 남기기 때문이다.
HIGH_52W_PERIOD = 250        # 52주 고가를 볼 창의 길이
HIGH_52W_GAP_DAYS = 15       # 창을 오늘에서 이만큼 띄운다 -> [i-265, i-15)
BREAKOUT_THRESHOLD = 1.015   # 종가 > 52주 고가 x 1.015 (1.5% 이상 확실히 넘김)
MA150_PERIOD = 150           # 추세 필터: 종가 > 150일선
MA150_SLOPE_DAYS = 20        # 그리고 150일선이 20일 전보다 높을 것(우상향)
MIN_MARKET_CAP = 1e11        # 시가총액 1,000억 (시점 기준)

VOLUME_SURGE_MULT = 1.5      # (구 정의 전용) 당일 거래량 / 50일 평균
MIN_TRADING_VALUE = 1e10     # (구 정의 전용) 거래대금 100억
MIN_BAR_INDEX = 285          # [i-265,i-15) 창 + 150일선 여유
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

# --- 급등주 연구 (EVENT_MODE="surge") ---
# 52주 신고가 연구에서 상한가가 알파의 대부분을 만들고 그 상한가는 종가에
# 살 수 없다는 것이 확인된 뒤(리포트 §5-5), 상한가를 배제한 급등주로 축을
# 옮겼다. 임계값은 민감도 스캔으로 정했다 — 등락률 하한을 5%~20% 어디에
# 두든 기저가 음수이고 승률이 39~40%로 고정이라, 체리피킹 여지가 없는
# 자리임을 확인한 뒤 표본이 가장 큰 관용적 값(10%)을 골랐다.
SURGE_MIN_CHANGE = 0.10       # 당일 종가 등락률 하한
SURGE_MIN_TRADING_VALUE = 1e9   # 당일 거래대금 10억
SURGE_MIN_MARKET_CAP = 1e11     # 시가총액 1,000억 (시점 기준)
# 거래량 조건은 걸지 않는다: 넣어도 기저가 -0.560% -> -0.631%로 나빠질 뿐
# 결론이 바뀌지 않아, 자유도를 하나 덜 만드는 쪽을 택했다.

EVENT_MODE = "high52"         # "high52" | "surge"

# --- 라벨 (스펙 §6) ---
COST = 0.0021                # 왕복 거래비용

# 청산 시점. 라벨의 의미를 바꾸는 설정이므로 데이터셋에 함께 기록하고,
# 불러올 때 현재 설정과 다르면 즉시 실패시킨다(dataset.load_dataset).
# 이름은 같은데 값의 정의가 다른 컬럼은 조용히 잘못된 결과를 만든다 —
# 다른 저장소에서 실제로 그 문제(학습 50봉 / 추론 20봉)를 확인했으므로,
# 여기서는 불일치가 반드시 예외로 드러나게 한다.
#   "open"  : 종가 매수 -> 익일 시가 청산 (52주 신고가 연구)
#   "close" : 종가 매수 -> 익일 종가 청산 (급등주 연구)
EXIT = "open"

# --- 검증 (스펙 §7.2) ---
EMBARGO_DAYS = 5
RANDOM_SEED = 42

# --- 경로 ---
LIST_SHARES_PATH = PROJECT_ROOT / "data" / "list_shares.parquet"   # 시점 상장주식수
HOLLYKR_CACHE = external("ohlcv_cache_dir")
CACHE_PATH = PROJECT_ROOT / "data" / "ohlcv_cache.pkl"
CATALYST_CSV = external("catalyst.csv")
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.parquet"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs"
