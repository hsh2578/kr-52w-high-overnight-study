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

  **재개봉(halt→resume) 오염 방지(코드리뷰 지적, 수정됨)**: np.isfinite만으로는
  부족하다. mark_halted는 정지봉의 Open을 NaN으로 바꾸지만 Close는 직전값을
  그대로 들고 있다(액면분할·감자 같은 자본 변경으로 정지되었다가 재개되면,
  재개봉의 Open은 새 가격 기준의 진짜 값이라 유한하다). 그 결과
  Open[t]/Close[t-1]에서 t-1이 정지봉이면 분모가 "정지 전 구가"의 낡은
  가격이 되어, 재개봉에서 수백~수천 %짜리 가짜 갭이 유한값으로 계산되고
  isfinite 필터를 통과해버린다(실사례: 003060, 2026-04-13~05-07 정지,
  Close가 251에 고정된 채 05-08 Open=4000 → +1494% "갭"). 그래서 갭 항은
  t-1(분모, 전일 종가)이 halted 봉이면 무조건 제외한다 — t(당일 Open)가
  halted면 이미 NaN이라 isfinite가 걸러내므로 별도 처리가 필요 없다.
  `halted` 컬럼이 있으면 그것을 쓰고(NaN 여부로 추론하지 않는다 — Close는
  정지봉에서도 NaN이 아니므로 추론이 불가능), 컬럼이 없으면(합성 테스트
  데이터 등) 전부 비정지로 간주한다.

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
    with np.errstate(divide="ignore", invalid="ignore"):
        # Close[t-1]<=0(방어적 경계)이면 0-나눗셈 경고가 뜨지만 결과는
        # inf/NaN이 되어 바로 아래 isfinite 필터에서 걸러지므로 무해하다.
        gaps_raw = (o.iloc[i - 19:i + 1].to_numpy() /
                    c.iloc[i - 20:i].to_numpy()) - 1.0
    # t-1(분모)이 정지봉이면 Close[t-1]이 낡은 가격이라 갭이 오염된다(위
    # docstring 참조) — halted 컬럼으로 그 항을 무조건 제외한다. t(당일
    # Open)가 정지봉인 경우는 이미 NaN이라 isfinite가 걸러낸다.
    if "halted" in df.columns:
        denom_halted = df["halted"].iloc[i - 20:i].to_numpy(dtype=bool)
    else:
        denom_halted = np.zeros(20, dtype=bool)
    gaps = gaps_raw[np.isfinite(gaps_raw) & ~denom_halted]
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
    # 이 창(t-1 = i-6..i-2)은 HALT_LOOKBACK이 보장하는 무정지 구간(i-4..i+1)
    # 밖이라, avg_gap_20d와 같은 이유로 t-1이 halted면 Close[t-1]이 낡은
    # 가격이 되어 재개봉에서 가짜 "급등"이 잡힐 수 있다(실사례: 086960
    # i=1499, halted 재개 직후 하루 수익률이 +150%로 계산되어 급등 오탐).
    # 그래서 같은 원칙으로 t-1이 halted인 항은 제외한다.
    prior_raw = c.iloc[i - 5:i].to_numpy() / c.iloc[i - 6:i - 1].to_numpy() - 1.0
    if "halted" in df.columns:
        prior_denom_halted = df["halted"].iloc[i - 6:i - 1].to_numpy(dtype=bool)
    else:
        prior_denom_halted = np.zeros(5, dtype=bool)
    prior = prior_raw[~prior_denom_halted]
    # 창의 전부가 정지 직후라 유효 비교가 하나도 없으면(현재 데이터에서는
    # 발생하지 않음) "급등 증거 없음"으로 보아 0.0을 반환한다 — 이 피처는
    # avg_gap_20d 같은 연속 평균이 아니라 0/1 지시자라, 결측을 NaN으로
    # 구분해도 얻는 정보가 없다(0은 원래도 "증거 없음"과 "확인상 급등
    # 없음"을 함께 나타낸다).
    prior_surge_5d = 1.0 if (prior.size and np.any(prior >= SURGE_LEVEL)) else 0.0

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
