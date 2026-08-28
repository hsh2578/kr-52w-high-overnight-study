# -*- coding: utf-8 -*-
"""F2 — D+1(익일 시가) 예측에 특화된 신규 피처 10개. 스펙 §5.2.

전 프로젝트(F1, stock-screener-kr 이식)에는 없던, 이 연구의 핵심 가설을
검증하기 위한 피처들이다: "예측 지평(익일 시가)이 바뀌면 필요한 피처도
바뀐다". 익일 시가 갭(overnight gap) 자체의 성향·당일 캔들 형태·요일·
정배열 여부처럼, D+N(수 주~수개월) 예측에서는 노이즈로 묻히지만 D+1
예측에는 직접적으로 관련되는 신호를 담는다.

NaN(거래정지 봉) 처리 방침
--------------------------
sources.mark_halted 에 의해 거래정지 봉은 Open/High/Low/Volume 이 NaN이다
(Close 는 직전값 유지). events.is_breakout 은 이벤트 당일(i)과 그 다음날
(i+1)의 정지만 배제한다(HALT_LOOKBACK=5는 i-4..i+1 구간만 본다) — 그보다
앞선 i-19..i-5 구간에는 정지봉이 얼마든지 섞일 수 있다.

avg_gap_20d/gap_vol_20d 는 20봉 창에서 Open[t]/Close[t-1] 을 구하므로,
정지봉의 NaN Open이 이 계산에 그대로 들어온다. 원칙:
  1. np.isfinite 로 걸러 유효한 갭만으로 집계한다(0으로 조용히 메우지
     않는다 — 0은 "평균 갭 0%"라는 별개의 주장이 되어버린다).
  2. 창 전체가 NaN이면(유효 갭 0개) 결과도 NaN을 반환한다 — 절대 0.0으로
     떨어뜨리지 않는다.
  3. 유효 갭이 1개뿐이면 표본분산이 정의되지 않지만, "결측"이 아니라
     "변동 관측치 없음"에 가까우므로 gap_vol_20d=0.0(모집단 표준편차,
     ddof=0 관례)을 반환한다. 이미 NaN을 걸러낸 배열에 대해 .std()를
     쓰므로(=np.nanstd가 아님) size==1이어도 RuntimeWarning이 나지 않는다.

market_gap_prev 도 같은 원칙이다: build_market_index는 그날 계산 가능한
종목이 0개(n==0)면 gap을 NaN으로 남긴다(sources.py 참조) — 이 역시 0.0으로
대체하지 않고 NaN을 그대로 전달한다.

intraday_range/upper_shadow 는 당일(i) High/Low를 쓴다. 이벤트 당일은
정지가 배제되므로 실데이터에서는 거의 항상 유한하지만, 방어적으로 NaN이면
0.0이 아니라 NaN을 반환한다(High==Low인 진짜 0-레인지와 결측을 구분).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

F2_NAMES = [
    "avg_gap_20d", "gap_vol_20d", "gap_open", "day_change",
    "intraday_range", "upper_shadow", "market_gap_prev", "dow",
    "aligned", "prior_surge_5d",
]

SURGE_LEVEL = 0.12   # '급등' 기준 (모의투자대회 backtest.py와 동일)


def compute_f2(df: pd.DataFrame, i: int, market: pd.DataFrame) -> dict:
    """봉 i의 F2 피처 10개."""
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    prev_close = float(c.iloc[i - 1])
    price = float(c.iloc[i])

    # 종목별 갭 성향: 최근 20봉(당일 포함, t=i-19..i)의 (Open[t]/Close[t-1] - 1)
    # 정렬: o.iloc[i-19:i+1] 의 k번째 원소는 Open[i-19+k],
    #       c.iloc[i-20:i]   의 k번째 원소는 Close[i-20+k] = Close[(i-19+k)-1]
    # 이므로 elementwise 나눗셈이 정확히 t, t-1 쌍이 된다. 마지막 원소
    # (k=19)는 Open[i]/Close[i-1] = 오늘의 gap_open 그 자체다.
    gaps = (o.iloc[i - 19:i + 1].to_numpy() /
            c.iloc[i - 20:i].to_numpy()) - 1.0
    gaps = gaps[np.isfinite(gaps)]
    if gaps.size == 0:
        avg_gap_20d = float("nan")
        gap_vol_20d = float("nan")
    elif gaps.size == 1:
        avg_gap_20d = float(gaps[0])
        gap_vol_20d = 0.0
    else:
        avg_gap_20d = float(gaps.mean())
        gap_vol_20d = float(gaps.std())  # ddof=0(모집단), 필터링 후라 경고 없음

    gap_open = (float(o.iloc[i]) - prev_close) / prev_close if prev_close > 0 else float("nan")
    day_change = (price - prev_close) / prev_close if prev_close > 0 else float("nan")

    hi = float(h.iloc[i])
    lo = float(l.iloc[i])
    rng = hi - lo  # NaN if either is NaN (halted bar, defensive)
    if not np.isfinite(rng):
        intraday_range = float("nan")
        upper_shadow = float("nan")
    else:
        intraday_range = rng / prev_close if prev_close > 0 else float("nan")
        upper_shadow = (hi - price) / rng if rng > 0 else 0.0

    prev_date = df.index[i - 1]
    market_gap_prev = float("nan")
    if market is not None and prev_date in market.index:
        val = market.loc[prev_date, "gap"]
        market_gap_prev = float(val) if np.isfinite(val) else float("nan")

    ma20 = float(c.iloc[i - 19:i + 1].mean())
    ma60 = float(c.iloc[i - 59:i + 1].mean())
    aligned = 1.0 if (price > ma20 > ma60) else 0.0

    # 정렬: c.iloc[i-5:i] 의 k번째 원소는 Close[i-5+k],
    #       c.iloc[i-6:i-1] 의 k번째 원소는 Close[i-6+k] = Close[(i-5+k)-1]
    # 이므로 t=i-5..i-1(당일 제외 직전 5봉)의 일간수익률 5개를 얻는다.
    prior = c.iloc[i - 5:i].to_numpy() / c.iloc[i - 6:i - 1].to_numpy() - 1.0
    prior_surge_5d = 1.0 if np.any(prior >= SURGE_LEVEL) else 0.0

    date = df.index[i]
    return {
        "avg_gap_20d": avg_gap_20d,
        "gap_vol_20d": gap_vol_20d,
        "gap_open": gap_open,
        "day_change": day_change,
        "intraday_range": intraday_range,
        "upper_shadow": upper_shadow,
        "market_gap_prev": market_gap_prev,
        "dow": float(date.dayofweek),
        "aligned": aligned,
        "prior_surge_5d": prior_surge_5d,
    }
