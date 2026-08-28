# -*- coding: utf-8 -*-
"""F1 — 기존 프로젝트(stock-screener-kr)의 기술 피처 14개. 스펙 §5.1.

원본: stock-screener-kr/ml/src/collect_training_data.py:calculate_features()
계산식을 그대로 옮기되 기준 시점만 이벤트 봉 i로 통일한다. 재현이 목적이므로
"개선"하지 않는다.

NaN(거래정지 봉) 처리 방침
--------------------------
sources.mark_halted 에 의해 거래정지 봉은 Open/High/Low/Volume 이 NaN이다
(Close 는 직전값 유지). 5~220봉 창에 정지봉이 섞이는 것이 일상적이므로:

- 창 집계는 전부 pandas Series.mean()/.max()/.min() 을 쓴다 — 이들은
  기본으로 skipna=True 라 NaN을 건너뛰고 나머지 값으로 계산한다(넘파이
  버전과 달리 NaN을 전파하지 않는다).
- 0 또는 NaN으로 나누는 것을 막는 `if x > 0` 가드는 NaN에 대해서도
  안전하다 — NaN > 0 은 False라 자연히 폴백 분기로 빠진다.
- calculate_atr 의 rolling(14).mean() 만은 예외다: pandas rolling의
  기본 min_periods는 window 크기(14)와 같아서, 14봉 창에 정지봉이 단
  하루만 섞여도(트루레인지가 NaN이 되어) 그 날짜의 ATR과 그 뒤 최대
  13영업일치 ATR까지 통째로 NaN이 되어버린다. 정지가 1.6%만 나와도
  ATR 결측률은 롤링 창 때문에 훨씬 커진다. 그래서 min_periods를
  window의 절반(7)으로 낮춰, 창 안에 정지봉이 섞여도(최대 7개까지)
  나머지 유효 TR만으로 ATR이 살아남게 한다. 카카오 액면분할처럼 3일
  연속 정지가 껴도 min_periods=7 이면 문제없이 계산된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from surge_d1 import config

F1_NAMES = [
    "breakout_pct", "volume_surge", "close_strength",
    "base_length", "volatility_contraction", "volume_dry_up",
    "ma200_slope", "pct_above_52w_low",
    "rs_vs_market", "market_return",
    "ma20_deviation", "liquidity",
    "days_since_ath", "atr_ratio",
]

_ATR_MIN_PERIODS_RATIO = 0.5  # window 절반까지는 정지봉이 섞여도 ATR 생존


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR.

    min_periods=period//2 로 낮춰서(위 모듈 docstring 참조) 창 안의
    고립된 정지봉 때문에 ATR 전체가 NaN으로 무너지지 않도록 한다.
    """
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    min_periods = max(1, int(period * _ATR_MIN_PERIODS_RATIO))
    return tr.rolling(period, min_periods=min_periods).mean()


def _market_return_20d(market: pd.DataFrame, date, i_fallback: int) -> float:
    """시장 20일 누적 수익률(%). market은 index=Date, 컬럼 ret(일간 수익률)."""
    if market is None or date not in market.index:
        return 0.0
    loc = market.index.get_loc(date)
    if isinstance(loc, slice) or loc < 20:
        return 0.0
    window = market["ret"].iloc[loc - 19:loc + 1].fillna(0.0)
    return float((np.prod(1.0 + window.to_numpy()) - 1.0) * 100)


def compute_f1(df: pd.DataFrame, i: int, market: pd.DataFrame) -> dict:
    """봉 i의 F1 피처 14개."""
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    price = float(close.iloc[i])
    P = config.HIGH_52W_PERIOD

    prior_high = float(close.iloc[max(0, i - P):i].max())
    breakout_pct = (price - prior_high) / prior_high * 100 if prior_high > 0 else 0.0

    vol50 = float(vol.iloc[max(0, i - 50):i].mean())
    volume_surge = float(vol.iloc[i]) / vol50 if vol50 > 0 else 0.0

    rng = float(high.iloc[i] - low.iloc[i])
    close_strength = (price - float(low.iloc[i])) / rng if rng > 0 else 0.5

    # 직전 52주 고가가 나온 위치 이후 경과일
    seg = close.iloc[max(0, i - P):i]
    base_length = int(i - (seg.index.get_loc(seg.idxmax()) + max(0, i - P))) if len(seg) else 0

    r5 = (float(high.iloc[i - 4:i + 1].max()) - float(low.iloc[i - 4:i + 1].min())) / price
    r20 = (float(high.iloc[i - 19:i + 1].max()) - float(low.iloc[i - 19:i + 1].min())) / price
    volatility_contraction = r5 / r20 if r20 > 0 else 1.0

    vol10 = float(vol.iloc[i - 9:i + 1].mean())
    volume_dry_up = vol10 / vol50 if vol50 > 0 else 1.0

    ma200_now = float(close.iloc[i - 199:i + 1].mean())
    ma200_prev = float(close.iloc[i - 219:i - 19].mean())
    ma200_slope = (ma200_now - ma200_prev) / ma200_prev if ma200_prev > 0 else 0.0

    low_52w = float(close.iloc[max(0, i - P):i + 1].min())
    pct_above_52w_low = (price - low_52w) / low_52w * 100 if low_52w > 0 else 0.0

    stock_ret_20d = (price / float(close.iloc[i - 20]) - 1) * 100
    market_return = _market_return_20d(market, df.index[i], i)
    rs_vs_market = stock_ret_20d - market_return

    ma20 = float(close.iloc[i - 19:i + 1].mean())
    ma20_deviation = (price / ma20 - 1) if ma20 > 0 else 0.0

    liquidity = float((close * vol).iloc[i - 4:i + 1].mean())

    ath_seg = close.iloc[:i]
    days_since_ath = int(i - ath_seg.index.get_loc(ath_seg.idxmax())) if len(ath_seg) else 0

    atr = calculate_atr(df, 14)
    atr_val = float(atr.iloc[i]) if np.isfinite(atr.iloc[i]) else 0.0
    atr_ratio = atr_val / price if price > 0 else 0.0

    return {
        "breakout_pct": breakout_pct,
        "volume_surge": volume_surge,
        "close_strength": close_strength,
        "base_length": float(base_length),
        "volatility_contraction": volatility_contraction,
        "volume_dry_up": volume_dry_up,
        "ma200_slope": ma200_slope,
        "pct_above_52w_low": pct_above_52w_low,
        "rs_vs_market": rs_vs_market,
        "market_return": market_return,
        "ma20_deviation": ma20_deviation,
        "liquidity": liquidity,
        "days_since_ath": float(days_since_ath),
        "atr_ratio": atr_ratio,
    }
