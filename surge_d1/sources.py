# -*- coding: utf-8 -*-
"""데이터 로더. 로컬 캐시만 읽고 네트워크를 쓰지 않는다.

pickle 사용 주의: 읽는 pkl은 1st-party 로컬 캐시다(HollyKR 프로젝트가 자기
머신에서 쓴 것, 또는 scripts/refresh_cache.py 가 쓴 것). 외부 파일을
역직렬화하지 않는다.
"""
from __future__ import annotations

import pickle
import re

import numpy as np
import pandas as pd

from surge_d1 import config

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def mark_halted(df: pd.DataFrame) -> pd.DataFrame:
    """거래정지 봉을 표시하고 Open/Volume을 NaN으로 바꾼다.

    원본(FDR/네이버)은 거래정지일을 Open=0·Volume=0·Close=직전값으로 채운다.
    예: 카카오(035720) 2021-04-15 액면분할 처리로 04-12~14 정지.
    이 값을 그대로 두면 라벨이 −100%가 되고 거래량 평균이 왜곡되므로 NaN 처리한다.
    Close는 직전값이 그대로 유효한 참조가이므로 건드리지 않는다.
    """
    out = df.copy()
    halted = (out["Open"].fillna(0) <= 0) | (out["Volume"].fillna(0) <= 0)
    out["halted"] = halted.to_numpy()
    out.loc[halted, "Open"] = np.nan
    out.loc[halted, "Volume"] = np.nan
    return out


def load_ohlcv() -> dict[str, pd.DataFrame]:
    """갱신본이 있으면 그것을, 없으면 HollyKR 원본을 읽는다.

    원본은 key가 "{code}_{days}" 형태라 code별 최장 시계열만 남긴다.
    반환 전에 mark_halted를 적용한다.
    """
    if config.CACHE_PATH.exists():
        with open(config.CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if cache and not re.match(r"\d+_\d+$", next(iter(cache))):
            return {k: mark_halted(v[OHLCV_COLS])
                    for k, v in cache.items() if v is not None}
    else:
        path = config.HOLLYKR_CACHE / "ohlcv" / "ohlcv_cache.pkl"
        with open(path, "rb") as f:
            cache = pickle.load(f)

    best: dict[str, pd.DataFrame] = {}
    for key, df in cache.items():
        m = re.match(r"(\d+)_(\d+)$", str(key))
        code = m.group(1) if m else str(key)
        if df is None or len(df) == 0:
            continue
        if code not in best or len(df) > len(best[code]):
            best[code] = df
    return {c: mark_halted(d[OHLCV_COLS]) for c, d in best.items()}


def load_catalyst() -> pd.DataFrame:
    """재료 CSV → 정규화된 DataFrame.

    반환 컬럼: date, code, ctype, reason, day_chg, d1
    """
    raw = pd.read_csv(config.CATALYST_CSV, encoding="utf-8")
    raw.columns = ["date", "name", "code", "chg_s", "day_chg",
                   "ctype", "reason", "d1", "d2", "d3", "cum3"]
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["date"]),
        "code": raw["code"].astype(str).str.zfill(6),
        "ctype": raw["ctype"].fillna("").astype(str).str.strip(),
        "reason": raw["reason"].fillna("").astype(str).str.strip(),
        "day_chg": pd.to_numeric(raw["day_chg"], errors="coerce"),
        "d1": pd.to_numeric(raw["d1"], errors="coerce"),
    })
    return out.reset_index(drop=True)


def build_market_index(ohlcv: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """전 종목 등가중 시장 지수. FDR의 KS11 호출이 불안정해 대용으로 쓴다.

    반환: index=Date, columns=[ret, gap, n]
      ret = 종목별 (Close/Close[-1] - 1) 의 그날 평균
      gap = 종목별 (Open/Close[-1] - 1) 의 그날 평균
      n   = 그날 계산에 쓰인 종목 수
    """
    rets, gaps = [], []
    for df in ohlcv.values():
        c = df["Close"].astype(float)
        o = df["Open"].astype(float)
        prev = c.shift(1)
        rets.append((c / prev - 1.0).rename(None))
        gaps.append((o / prev - 1.0).rename(None))
    r = pd.concat(rets, axis=1)
    g = pd.concat(gaps, axis=1)
    out = pd.DataFrame({
        "ret": r.mean(axis=1, skipna=True),
        "gap": g.mean(axis=1, skipna=True),
        "n": r.count(axis=1).astype(int),
    }).sort_index()
    out.loc[out["n"] == 0, ["ret", "gap"]] = np.nan
    return out
