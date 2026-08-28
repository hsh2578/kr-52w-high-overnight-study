# -*- coding: utf-8 -*-
"""52주 신고가 돌파 이벤트 스크린과 D+1 라벨. 스펙 §3.1, §6.

pickle 사용 주의: 읽는 krx_master_*.pkl 은 HollyKR 프로젝트가 자기 머신에서 쓴
1st-party 로컬 캐시다. 외부에서 받은 파일을 역직렬화하지 않는다.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from surge_d1 import config


def is_breakout(df: pd.DataFrame, i: int) -> bool:
    """봉 i가 52주 신고가 돌파 이벤트인가.

    조건 (스펙 §3.1): 신고가 · 거래량 1.5배 · 거래대금 100억 · i>=260 · 익일 존재
    · 최근 HALT_LOOKBACK봉과 익일 봉에 거래정지 없음
    """
    if i < config.MIN_BAR_INDEX or i + 1 >= len(df):
        return False

    # 거래정지 배제: 정지 구간은 Open/Volume이 NaN이라 지표가 왜곡된다
    if "halted" in df.columns:
        lo = max(0, i - config.HALT_LOOKBACK + 1)
        if bool(df["halted"].iloc[lo:i + 2].any()):
            return False

    c = float(df["Close"].iloc[i])
    v = float(df["Volume"].iloc[i])
    if not np.isfinite(c) or not np.isfinite(v) or c <= 0:
        return False

    # 가격제한 위반 봉 배제: 캐시에 섞인 오류 데이터(예: 037460 2026-05-15,
    # 계산상 +32.05%이나 재료 CSV엔 +29.97%로 기록됨)를 이벤트로 보지 않는다.
    prev_c = float(df["Close"].iloc[i - 1])
    if np.isfinite(prev_c) and prev_c > 0:
        if abs(c / prev_c - 1.0) > config.MAX_DAILY_MOVE:
            return False

    prior_high = float(df["Close"].iloc[i - config.HIGH_52W_PERIOD:i].max())
    if not (c > prior_high):
        return False

    # NaN(정지봉)을 건너뛴 평균
    vol_avg = float(df["Volume"].iloc[i - 50:i].mean())
    if not (np.isfinite(vol_avg) and vol_avg > 0
            and v >= config.VOLUME_SURGE_MULT * vol_avg):
        return False

    return bool(c * v >= config.MIN_TRADING_VALUE)


def labels(df: pd.DataFrame, i: int, cost: float = config.COST) -> dict:
    """봉 i 종가 매수 기준 라벨. 스펙 §6.

    y_raw    = (Open[i+1]-Close[i])/Close[i] - cost    (익일 시가청산, 비용차감)
    y_hold   = (Close[i+1]-Close[i])/Close[i] - cost   (익일 종가청산, 비용차감)
    y_hold20 = (Close[i+20]-Close[i])/Close[i] - cost  (20봉 보유, 비용차감)
               i+20이 시계열 끝을 넘어가면 NaN — 그렇다고 이벤트 자체를 버리면
               최근 한 달치(라이브 데모에서 가장 보고 싶은 구간)가 통째로
               사라지므로, y_hold20만 NaN으로 두고 이벤트는 그대로 남긴다.
    y_cls    = 1 if y_raw > 0 else 0                   (학습 타깃, 비용차감 기준)
    y_up     = 1 if Open[i+1] > Close[i] else 0         (집계 전용, 비용 미차감)
               y_cls와 y_up이 둘 다 필요한 이유: 0.1% 오른 거래는 비용(0.21%)을
               못 이기고 실제로는 손실이므로 학습 타깃(y_cls)은 비용을 반드시
               차감해야 한다. 반면 y_up은 "익일 상승 확률 X% vs 익일 순이익
               확률 Y%"를 리포트에서 나란히 보여주기 위한 값으로, 그 자체가
               학습에 쓰이지는 않는다 — 둘의 차이(X-Y)가 이 전략의 실질적인
               발견이므로 어느 한쪽으로 "단순화"해 없애면 안 된다.
    """
    c = float(df["Close"].iloc[i])
    o1 = float(df["Open"].iloc[i + 1])
    c1 = float(df["Close"].iloc[i + 1])
    y_raw = (o1 - c) / c - cost if (c > 0 and o1 > 0) else np.nan
    y_hold = (c1 - c) / c - cost if (c > 0 and c1 > 0) else np.nan
    y_up = int(o1 > c) if (c > 0 and o1 > 0) else 0

    if i + 20 < len(df):
        c20 = float(df["Close"].iloc[i + 20])
        y_hold20 = (c20 - c) / c - cost if (c > 0 and c20 > 0) else np.nan
    else:
        y_hold20 = np.nan

    return {
        "y_raw": y_raw,
        "y_hold": y_hold,
        "y_hold20": y_hold20,
        "y_cls": int(y_raw > 0) if np.isfinite(y_raw) else 0,
        "y_up": y_up,
    }


def load_excluded_codes() -> set[str]:
    """제외할 코드(우선주 등 비보통주) 집합.

    HollyKR 마스터의 `is_common` 만 쓴다. `is_etf` 는 신뢰할 수 없다 —
    삼성전자(005930)가 True 로 잘못 표시돼 있음을 실측 확인했다.
    마스터에 없는 코드(신규상장·상장폐지)는 제외하지 않는다: 상장폐지 종목을 남기는 것이
    생존편향을 줄인다.
    """
    masters = sorted(config.HOLLYKR_CACHE.glob("krx_master_*.pkl"))
    if not masters:
        return set()
    try:
        with open(masters[-1], "rb") as f:
            m = pickle.load(f)
    except Exception:
        return set()
    codes = m["종목코드"].astype(str).str.zfill(6)
    return set(codes[~m["is_common"].astype(bool)])


def collect_events(ohlcv: dict[str, pd.DataFrame],
                   exclude: set[str] | None = None) -> pd.DataFrame:
    """전 종목 순회하며 이벤트와 라벨을 모은다.

    exclude 에 든 코드는 건너뛴다(우선주 등). None 이면 필터를 걸지 않는다.
    """
    rows = []
    for code, df in ohlcv.items():
        if exclude and code in exclude:
            continue
        n = len(df)
        if n < config.MIN_BAR_INDEX + 2:
            continue
        c = df["Close"].to_numpy(dtype=float)
        v = df["Volume"].to_numpy(dtype=float)
        tval = c * v
        for i in range(config.MIN_BAR_INDEX, n - 1):
            if tval[i] < config.MIN_TRADING_VALUE:
                continue
            if not is_breakout(df, i):
                continue
            lab = labels(df, i)
            if not np.isfinite(lab["y_raw"]):
                continue
            rows.append({"code": code, "date": df.index[i], "i": i,
                         "close": c[i], **lab})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["date", "code"]).reset_index(drop=True)
    return out


def add_market_adjusted(events: pd.DataFrame) -> pd.DataFrame:
    """같은 날 이벤트들의 y_raw 중앙값을 빼 시장조정 라벨을 만든다. 스펙 §6."""
    out = events.copy()
    med = out.groupby("date")["y_raw"].transform("median")
    out["y_mkt"] = out["y_raw"] - med
    return out
