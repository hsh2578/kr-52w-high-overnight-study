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
    """거래정지 봉을 표시하고 Open/High/Low/Volume을 NaN으로 바꾼 뒤,
    남은 정상 봉의 고저 불변식을 복구한다.

    원본(FDR/네이버)은 거래정지일을 Open=High=Low=0·Volume=0·Close=직전값으로
    채운다. 예: 099520 2021-03-26, 카카오(035720) 2021-04-12~14(액면분할 처리).
    Open·Volume만 보고 정지를 판정하고 High/Low를 0으로 남겨두면 라벨이
    −100%가 되고 거래량 평균이 왜곡될 뿐 아니라, 후속 피처가 20봉 안팎의
    창을 되돌아볼 때 min(Low)=0·max(High)=0 취급되어 RSV·ATR·변동성 압축
    등이 오염된다. 그래서 Open/High/Low/Volume 중 하나라도 0 이하(또는 NaN)면
    정지로 보고 네 값을 모두 NaN으로 바꾼다.
    Close는 직전값이 그대로 유효한 참조가이므로 건드리지 않는다.

    고저 불변식 복구(정지봉 제외):
    정의상 그날의 High는 그날 거래된 모든 가격 이상이어야 하고 Low는
    이하여야 한다. 그런데 실측 캐시(약 440만 봉)에서 약 9,183봉(0.208%)이
    High < max(Open,Close) 또는 Low > min(Open,Close)를 어긴다 — 80.5%는
    1원 차이, 96.1%는 5원 이내로, 분할·액면가 조정 시 정수 반올림 과정에서
    생긴 오차로 보인다. 이는 계산식을 "개선"하는 게 아니라 정의상 있을 수
    없는 값을 고치는 것이다: Open/Close는 실제 체결·동시호가 가격이라
    원본 그대로 신뢰하고, High/Low는 그로부터 도출되는 포락선이므로 포락선이
    실제 체결가와 모순되면 포락선 쪽이 틀린 것이다. 이 정정을 하지 않으면
    close_strength=(Close-Low)/(High-Low), upper_shadow=(High-Close)/(High-Low),
    RSV_20, KMID2 등 [0,1] 범위로 정의된 피처들이 그 범위를 벗어나
    (예: close_strength 최대 1.02, upper_shadow 최소 −0.02) 설명하기 어려운
    이상치로 결과에 나타난다. 오차 규모가 미미하므로(80.5%가 1원) 이 복구가
    실제 신호를 왜곡하지는 않는다.
    정지봉은 이미 Open/High/Low가 NaN이므로 손대면 안 된다 — np.maximum/
    np.minimum은 (pandas의 skipna=True 집계와 달리) NaN을 만나면 NaN을
    그대로 반환해 전파하므로, 정지봉의 NaN High/Low가 Close 값으로
    되살아나지 않는다. 이는 명시적으로 테스트로 검증한다.
    """
    out = df.copy()
    halted = (
        (out["Open"].fillna(0) <= 0)
        | (out["High"].fillna(0) <= 0)
        | (out["Low"].fillna(0) <= 0)
        | (out["Volume"].fillna(0) <= 0)
    )
    out["halted"] = halted.to_numpy()
    out.loc[halted, ["Open", "High", "Low", "Volume"]] = np.nan

    o = out["Open"].to_numpy(dtype=float)
    h = out["High"].to_numpy(dtype=float)
    l = out["Low"].to_numpy(dtype=float)
    c = out["Close"].to_numpy(dtype=float)
    # np.maximum/np.minimum은 NaN과 비교하면 NaN을 반환하므로, 정지봉(Open·High가
    # NaN)은 Close가 유효해도 이 연산을 거쳐도 그대로 NaN으로 남는다.
    out["High"] = np.maximum(np.maximum(h, o), c)
    out["Low"] = np.minimum(np.minimum(l, o), c)
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


def load_list_shares() -> dict[str, pd.Series]:
    """종목별 **시점 상장주식수** 시계열. scripts/fetch_marketcap.py의 산출물.

    반환: {code: Series(index=스냅샷 날짜, values=상장주식수)} — 날짜 오름차순.

    쓰는 법:
        shares(t) = 시계열에서 t **이전**의 가장 가까운 스냅샷
        시가총액(t) = shares(t) x 종가(t)

    왜 '이전'인가: t 시점에 알 수 있는 최신 값만 써야 한다. 미래 스냅샷을
    쓰면 유상증자·분할을 미리 아는 셈이 된다.
    """
    if not config.LIST_SHARES_PATH.exists():
        raise FileNotFoundError(
            f"{config.LIST_SHARES_PATH} 없음. 먼저 실행: "
            "python scripts/fetch_marketcap.py")
    df = pd.read_parquet(config.LIST_SHARES_PATH)
    return {c: g.set_index("date")["shares"].sort_index()
            for c, g in df.groupby("code")}


def market_cap_at(shares: pd.Series | None, date, close: float) -> float:
    """시점 시가총액. 해당 날짜 이전 스냅샷이 없으면 NaN.

    NaN을 0이나 현재값으로 메우지 않는 이유: 그러면 상장 초기 종목이
    조용히 필터를 통과하거나(0) 미래 정보가 섞인다(현재값). 모르면
    모른다고 두고 이벤트에서 빼는 편이 정직하다.
    """
    if shares is None or not np.isfinite(close):
        return float("nan")
    pos = shares.index.searchsorted(pd.Timestamp(date), side="right") - 1
    if pos < 0:
        return float("nan")
    return float(shares.iloc[pos]) * float(close)


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
