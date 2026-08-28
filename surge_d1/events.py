# -*- coding: utf-8 -*-
"""52주 신고가 돌파 이벤트 스크린과 D+1 라벨. 스펙 §3.1, §6.

pickle 사용 주의: 읽는 krx_master_*.pkl 은 HollyKR 프로젝트가 자기 머신에서 쓴
1st-party 로컬 캐시다. 외부에서 받은 파일을 역직렬화하지 않는다.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from surge_d1 import config


def is_breakout(df: pd.DataFrame, i: int) -> bool:
    """봉 i가 52주 신고가 돌파 이벤트인가.

    조건 (스펙 §3.1): 252봉 종가 신고가 · 거래량이 **직전 50봉 평균**의 1.5배
    · 당일 거래대금 100억 · i>=260 · 익일 존재
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


def labels(df: pd.DataFrame, i: int, cost: float = config.COST,
           cfg_exit: str | None = None) -> dict:
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
    cfg_exit = config.EXIT if cfg_exit is None else cfg_exit
    if cfg_exit not in ("open", "close"):
        raise ValueError(f"EXIT는 'open' 또는 'close': {cfg_exit!r}")

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

    # 청산 시점(config.EXIT)에 따라 이 연구가 실제로 쓰는 라벨이 정해진다.
    #   "open"  -> y_raw  (종가 매수 -> 익일 시가)
    #   "close" -> y_hold (종가 매수 -> 익일 종가)
    # `y_exit`/`y_cls`가 파이프라인이 학습·평가에 쓰는 컬럼이고, y_raw/y_hold는
    # 항상 같은 뜻을 유지해 비교·검증에 쓴다. 이름이 같은데 뜻이 다른 컬럼을
    # 만들지 않기 위한 구분이다.
    y_exit = y_raw if cfg_exit == "open" else y_hold

    return {
        "y_raw": y_raw,
        "y_hold": y_hold,
        "y_hold20": y_hold20,
        "y_exit": y_exit,
        "y_cls": int(y_exit > 0) if np.isfinite(y_exit) else 0,
        "y_up": y_up,
    }


def load_excluded_codes() -> set[str]:
    """제외할 코드(개별 보통주가 아닌 것) 집합.

    1순위: `data/universe.json` (KRX 공식 API 기반). ETF·ETN·ELW·우선주·리츠·
    외국주권·DR·인프라펀드까지 포괄한다. HollyKR 마스터의 `is_etf` 플래그는
    신뢰할 수 없다 — 삼성전자(005930)가 True 로 잘못 표시돼 있음을 실측
    확인했다. universe.json이 없으면(또는 읽기 실패하면) HollyKR 마스터의
    `is_common` 로 폴백하고, 그마저 없으면 빈 집합을 돌려준다 — 파일이 없다는
    이유만으로 예외를 던지지 않는다.

    어느 목록에도 없는 코드(신규상장·상장폐지 등)는 제외하지 않는다:
    "제외 목록에 명시된 것만 뺀다"는 규칙이 상장폐지 종목을 남겨 생존편향을
    줄인다. (반대로 "보통주 목록에 있어야만 포함"으로 뒤집으면 안 된다 —
    collect_events의 exclude 시맨틱과 어긋난다.)
    """
    universe_path = config.PROJECT_ROOT / "data" / "universe.json"
    if universe_path.exists():
        try:
            with open(universe_path, encoding="utf-8") as f:
                data = json.load(f)
            return {str(c) for c in data.get("excluded", [])}
        except Exception:
            pass  # 손상된 파일이면 아래 마스터 폴백으로 넘어간다

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


def is_surge(df: pd.DataFrame, i: int) -> bool:
    """봉 i가 급등주 이벤트인가 (거래대금·시가총액 조건은 여기서 보지 않는다).

    조건: 당일 종가 등락률 >= SURGE_MIN_CHANGE · **상한가 제외** ·
    손상 데이터(MAX_DAILY_MOVE 초과) 배제 · i>=MIN_BAR_INDEX · 익일 존재 ·
    최근 HALT_LOOKBACK봉과 익일 봉에 거래정지 없음.

    상한가를 빼는 이유: 상한가는 종가 동시호가에서 매수 잔량만 쌓이고 매도가
    말라붙어 종가 체결을 가정할 수 없다. 52주 신고가 연구에서 모델 픽의 74%가
    상한가였고 그것이 성과 대부분을 만들고 있었다(리포트 §5-5) — 살 수 없는
    거래의 수익을 성과로 보고하지 않기 위해 이벤트 단계에서 배제한다.

    거래대금은 조건이 아니라 **피처**로 둔다(F6 turnover_today). 실측상
    거래대금이 클수록 되돌림이 강한 단조 관계(t=-9.30)가 있어, 잘라내면
    가장 강한 변수 하나를 버리게 된다.
    """
    if i < config.MIN_BAR_INDEX or i + 1 >= len(df):
        return False

    if "halted" in df.columns:
        lo = max(0, i - config.HALT_LOOKBACK + 1)
        if bool(df["halted"].iloc[lo:i + 2].any()):
            return False

    c = float(df["Close"].iloc[i])
    prev_c = float(df["Close"].iloc[i - 1])
    if not (np.isfinite(c) and np.isfinite(prev_c) and c > 0 and prev_c > 0):
        return False

    chg = c / prev_c - 1.0
    if chg > config.MAX_DAILY_MOVE:      # 물리적으로 불가능한 값 = 손상 데이터
        return False
    return bool(config.SURGE_MIN_CHANGE <= chg < config.LIMIT_UP_PCT)


def collect_surge_events(ohlcv: dict[str, pd.DataFrame],
                         shares: dict[str, pd.Series],
                         exclude: set[str] | None = None) -> pd.DataFrame:
    """급등주 이벤트와 라벨을 모은다.

    `shares`는 sources.load_list_shares()의 결과 — 시점 시가총액을 계산해
    SURGE_MIN_MARKET_CAP 미만을 거른다. 스냅샷이 없는 구간(상장 직후 등)은
    시총을 알 수 없으므로 이벤트에서 뺀다. 0으로 메우거나 현재 시총을
    끌어다 쓰면 각각 필터 무력화·미래정보가 된다.
    """
    from surge_d1 import sources

    rows = []
    for code, df in ohlcv.items():
        if exclude and code in exclude:
            continue
        n = len(df)
        if n < config.MIN_BAR_INDEX + 2:
            continue
        sh = shares.get(code)
        if sh is None:
            continue
        c = df["Close"].to_numpy(dtype=float)
        v = df["Volume"].to_numpy(dtype=float)
        idx = df.index
        for i in range(config.MIN_BAR_INDEX, n - 1):
            if not is_surge(df, i):
                continue
            mcap = sources.market_cap_at(sh, idx[i], c[i])
            if not np.isfinite(mcap) or mcap < config.SURGE_MIN_MARKET_CAP:
                continue
            lab = labels(df, i)
            if not np.isfinite(lab["y_exit"]):
                continue
            rows.append({"code": code, "date": idx[i], "i": i,
                         "close": c[i], "market_cap": mcap,
                         "tval_today": c[i] * v[i], **lab})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["date", "code"]).reset_index(drop=True)
    return out


def add_market_adjusted(events: pd.DataFrame) -> pd.DataFrame:
    """같은 날 이벤트들의 y_raw 중앙값을 빼 시장조정 라벨을 만든다. 스펙 §6."""
    out = events.copy()
    # 라벨과 같은 청산 시점을 써야 한다 — 시가청산 연구인데 종가 기준
    # 중앙값을 빼면 두 정의가 섞인다.
    med = out.groupby("date")["y_exit"].transform("median")
    out["y_mkt"] = out["y_exit"] - med
    return out
