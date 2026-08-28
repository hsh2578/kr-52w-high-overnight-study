# -*- coding: utf-8 -*-
"""F6 — 급등주 특화 피처 8개: 거래대금 · 저항선 · 과열도.

왜 필요한가. 기존 F1~F5는 **52주 신고가 돌파**를 전제로 설계됐다. 급등주
연구로 축을 옮기면서 세 가지가 비게 됐다.

1. **당일 거래대금.** F1의 `liquidity`는 5일 평균이라 "오늘 얼마나 이례적으로
   거래됐나"를 담지 못한다. 그런데 실측상 이것이 가장 강한 단일 신호다 —
   상한가를 뺀 급등주 26,343건에서 당일 거래대금 구간별 익일 종가청산 기저가
   0~10억 +0.159% / 10~100억 -0.150% / 100~500억 -0.326% / 500억+ -0.654%
   (t=-9.30)로 단조 하락한다. 거래대금이 큰 급등일수록 되돌림이 강하다.

2. **저항선.** "위에 물린 물량이 많을수록 급등이 되돌려진다"는 가설을 직접
   검정할 변수가 하나도 없었다. `overhead_supply_60d`가 그 핵심이다.

3. **과열도.** 급등 당일만 보면 "이미 며칠째 오르던 중인가"를 알 수 없다.

모든 피처는 봉 i **당일 종가까지의 정보만** 쓴다. i+1 이후를 참조하는 항은
하나도 없다 — 이 모듈의 모든 창은 `[..., i]` 로 닫힌다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

F6_NAMES = (
    "turnover_today", "turnover_ratio",
    "dist_to_60d_high", "overhead_supply_60d", "above_60d_high",
    "gain_5d", "gain_20d", "consec_up",
)

RESIST_WINDOW = 60      # 저항선을 볼 창(약 3개월). 급등 직전 매물대의 관례적 범위.


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def compute_f6(df: pd.DataFrame, i: int) -> dict:
    """봉 i의 F6 피처 8개.

    `df`는 Open/High/Low/Close/Volume을 가진 일봉 프레임, `i`는 이벤트 봉의
    위치(0-기반). 거래정지 봉은 sources.mark_halted가 이미 NaN 처리했으므로
    여기서는 nan-안전 집계(np.nanmax 등)로 건너뛴다.
    """
    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float)

    c = close[i]
    if not np.isfinite(c) or c <= 0:
        return {k: float("nan") for k in F6_NAMES}

    # --- 거래대금 ---
    tval_today = c * vol[i] if np.isfinite(vol[i]) else np.nan
    # 로그를 쓰는 이유: 거래대금은 수 억~수 조로 4자릿수 넘게 퍼져 있어
    # 원값 그대로는 트리 분할이 꼬리에 끌려간다.
    turnover_today = np.log1p(tval_today) if np.isfinite(tval_today) else np.nan

    lo20 = max(0, i - 20)
    tv20 = close[lo20:i] * vol[lo20:i]          # **당일 제외** 직전 20봉
    base = np.nanmean(tv20) if len(tv20) else np.nan
    turnover_ratio = (tval_today / base
                      if np.isfinite(base) and base > 0 and np.isfinite(tval_today)
                      else np.nan)

    # --- 저항선 ---
    lo = max(0, i - RESIST_WINDOW)
    h_win = high[lo:i]              # 당일을 뺀 직전 60봉의 고가
    v_win = vol[lo:i]
    if len(h_win) and np.isfinite(h_win).any():
        h_max = np.nanmax(h_win)
        dist_to_60d_high = (h_max - c) / c
        above_60d_high = 1.0 if c > h_max else 0.0
        # 위에 물린 물량: 고가가 현재 종가보다 높았던 봉의 거래량 비중.
        # 그 봉에서 산 사람 중 일부는 지금 손실 구간에 있고, 본전 근처에서
        # 팔 유인이 생긴다 — 매물대(overhead supply)의 표준 대리변수다.
        m = np.isfinite(h_win) & np.isfinite(v_win)
        if m.any() and np.nansum(v_win[m]) > 0:
            overhead = float(np.nansum(v_win[m][h_win[m] > c]) / np.nansum(v_win[m]))
        else:
            overhead = np.nan
    else:
        dist_to_60d_high = above_60d_high = overhead = np.nan

    # --- 과열도 ---
    def gain(k: int) -> float:
        j = i - k
        if j < 0 or not np.isfinite(close[j]) or close[j] <= 0:
            return np.nan
        return c / close[j] - 1.0

    gain_5d, gain_20d = gain(5), gain(20)

    # 연속 상승일수(당일 포함). 정지봉(NaN)을 만나면 거기서 끊는다.
    consec = 0
    for j in range(i, 0, -1):
        a, b = close[j], close[j - 1]
        if not (np.isfinite(a) and np.isfinite(b)) or a <= b:
            break
        consec += 1
        if consec >= 20:            # 상한을 둬 극단값이 트리를 지배하지 않게 한다
            break

    return {
        "turnover_today": _safe(turnover_today),
        "turnover_ratio": _safe(turnover_ratio),
        "dist_to_60d_high": _safe(dist_to_60d_high),
        "overhead_supply_60d": _safe(overhead),
        "above_60d_high": _safe(above_60d_high),
        "gain_5d": _safe(gain_5d),
        "gain_20d": _safe(gain_20d),
        "consec_up": float(consec),
    }
