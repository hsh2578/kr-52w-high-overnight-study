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


def _dedup_longest(cache: dict) -> dict[str, pd.DataFrame]:
    """캐시 항목을 code 단위로 묶어 가장 긴 시계열만 남긴다.

    key 형태는 항목마다 개별 판단한다("{code}_{days}"면 접미사를 벗기고,
    아니면 그대로 code로 쓴다) — 캐시 전체의 형태를 첫 key 하나만 보고
    판단하면 혼재된 캐시(플레인 key와 접미사 key가 섞인 경우)에서
    접미사가 안 벗겨진 채로 새어나갈 수 있기 때문이다.
    None이거나 빈 DataFrame인 항목은 건너뛴다.
    """
    best: dict[str, pd.DataFrame] = {}
    for key, df in cache.items():
        m = re.match(r"(\d+)_(\d+)$", str(key))
        code = m.group(1) if m else str(key)
        if df is None or len(df) == 0:
            continue
        if code not in best or len(df) > len(best[code]):
            best[code] = df
    return best


def load_ohlcv() -> dict[str, pd.DataFrame]:
    """갱신본이 있으면 그것을, 없거나 쓸 항목이 없으면 HollyKR 원본을 읽는다.

    key 형태 판단과 code별 최장 시계열 선택은 _dedup_longest가 항목별로
    처리한다. 갱신본이 존재하더라도 언피클 결과가 빈 dict이거나 유효한
    항목이 하나도 없으면(None/빈 DataFrame뿐이면) 원본으로 대체한다.
    반환 전에 mark_halted를 적용한다.
    """
    best: dict[str, pd.DataFrame] = {}
    if config.CACHE_PATH.exists():
        with open(config.CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        best = _dedup_longest(cache)

    if not best:
        path = config.HOLLYKR_CACHE / "ohlcv" / "ohlcv_cache.pkl"
        with open(path, "rb") as f:
            cache = pickle.load(f)
        best = _dedup_longest(cache)

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
