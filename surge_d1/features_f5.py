# -*- coding: utf-8 -*-
"""F5 — 투자자 수급 피처 6개.

출처: scripts/collect_investor.py 가 키움 ka10059로 수집한 data/investor.pkl.
한국 시장의 익일 수익률 예측에서 수급은 가장 유력한 후보 변수지만, 전 기간을 주는
경로가 키움뿐이었다(KRX 404 · KIS 30일 · 로컬캐시 3개월).

pickle 사용 주의: 이 프로젝트가 직접 수집해 쓴 1st-party 파일이다.

단위: 수집 시 unit_tp=1000 이라 순매수는 천 단위다. 종목 규모 차이가 수십~수억 배라
절대값을 그대로 쓰면 안 되고, 반드시 거래대금으로 나눠 무차원화한다.
결측은 0이 아니라 NaN으로 둔다 - 0은 "순매수가 정확히 0"이라는 다른 정보다.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from surge_d1 import config

F5_NAMES = ["foreign_net_1d", "foreign_net_5d", "inst_net_1d",
            "inst_net_5d", "indiv_net_5d", "foreign_streak"]

NAN = {k: float("nan") for k in F5_NAMES}
STREAK_CLIP = 10


def load_investor() -> dict[str, pd.DataFrame]:
    """code → 수급 이력. 파일이 없으면 빈 dict(수급 없이 진행)."""
    path = config.PROJECT_ROOT / "data" / "investor.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        store = pickle.load(f)
    out = {}
    for code, df in store.items():
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        out[code] = d.sort_values("date").reset_index(drop=True)
    return out


def _ratio(num: float, den: float) -> float:
    """거래대금 대비 비율. 분모가 0/결측이면 NaN."""
    if not np.isfinite(den) or den == 0:
        return float("nan")
    return float(num) / float(den)


def compute_f5(inv: pd.DataFrame | None, date: pd.Timestamp) -> dict:
    """조회일(date) **이하** 구간만 보고 수급 피처를 만든다.

    date 이후 행은 미래정보이므로 반드시 잘라낸다.
    """
    if inv is None or len(inv) == 0:
        return dict(NAN)

    past = inv[inv["date"] <= date]
    if len(past) == 0:
        return dict(NAN)
    # 조회일 당일 데이터가 없으면(휴장·결측) 1일 피처는 NaN
    same_day = past[past["date"] == date]

    out = dict(NAN)

    if len(same_day):
        r = same_day.iloc[-1]
        out["foreign_net_1d"] = _ratio(r["foreign"], r["trd_value"])
        out["inst_net_1d"] = _ratio(r["inst"], r["trd_value"])

    w = past.tail(5)
    den5 = float(w["trd_value"].sum())
    if len(w) and np.isfinite(den5) and den5 > 0:
        out["foreign_net_5d"] = float(w["foreign"].sum()) / den5
        out["inst_net_5d"] = float(w["inst"].sum()) / den5
        out["indiv_net_5d"] = float(w["indiv"].sum()) / den5

    # 연속 순매수(양수) / 순매도(음수) 일수
    vals = past["foreign"].to_numpy(dtype=float)[::-1]
    streak = 0
    if len(vals) and np.isfinite(vals[0]) and vals[0] != 0:
        sign = 1 if vals[0] > 0 else -1
        for v in vals:
            if not np.isfinite(v) or v == 0 or (1 if v > 0 else -1) != sign:
                break
            streak += 1
        streak *= sign
    out["foreign_streak"] = float(np.clip(streak, -STREAK_CLIP, STREAK_CLIP))
    return out
