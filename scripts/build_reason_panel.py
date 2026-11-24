# -*- coding: utf-8 -*-
"""급등 사유 문장 + 익일 시가 수익률 패널.

대상: 그날의 급등주(특징주 정리 서비스의 일일 상위 20종목 + 텔레그램 '상한가 및 급등주').
라벨: 당일 종가 매수 → 익일 시가 매도, 왕복비용 0.21% 차감 (이 저장소의 y_raw와 같다).

같은 (날짜, 종목)이 두 출처에 다 있으면 정리 서비스 문장을 쓴다(기간이 길어 문체가 일관된다).

가격은 로컬 일봉 캐시(~2026-08-27)를 먼저 쓰고, 그 뒤 구간과 캐시에 없는 종목만
FinanceDataReader로 받는다. 받은 것은 data/reason_px.pkl 에 쌓아 다시 받지 않는다.

출력: data/reason_panel.parquet
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surge_d1 import config, sources  # noqa: E402

TG = config.PROJECT_ROOT / "data" / "tg_reasons.parquet"
PX = config.PROJECT_ROOT / "data" / "reason_px.pkl"     # 자체 생성 로컬 캐시
OUT = config.PROJECT_ROOT / "data" / "reason_panel.parquet"
NO_REASON = ("사유 없음", "미노출", "이유 없음", "특별한 이슈 없", "뚜렷한 이유")


def load_rows() -> pd.DataFrame:
    w = pd.read_csv(config.CATALYST_CSV, dtype={"코드": str})
    w = pd.DataFrame({"date": pd.to_datetime(w["날짜"]), "code": w["코드"],
                      "name": w["종목명"], "chg": w["당일등락pct"],
                      "reason": w["사유"].astype(str), "src": "weare"})
    t = pd.read_parquet(TG)
    t = t[t["kind"] == "surge_list"]
    t = pd.DataFrame({"date": t["date"], "code": t["code"], "name": t["name"],
                      "chg": t["chg"], "reason": t["reason"].astype(str), "src": "taja"})
    # 정리 서비스 자료에 코드가 없는 신규 상장 종목은 텔레그램의 이름·코드 쌍으로 메운다
    n2c = dict(t.dropna(subset=["code"])[["name", "code"]].drop_duplicates("name").values)
    w["code"] = w["code"].fillna(w["name"].map(n2c))
    df = pd.concat([w, t], ignore_index=True).dropna(subset=["code"])
    df["code"] = df["code"].str.zfill(6)
    df = df[~df["reason"].str.contains("|".join(NO_REASON))]
    return (df.sort_values(["date", "code", "src"], ascending=[True, True, False])
              .drop_duplicates(["date", "code"]).reset_index(drop=True))


def load_prices(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    import FinanceDataReader as fdr

    cache = sources.load_ohlcv()
    extra: dict = pickle.load(open(PX, "rb")) if PX.exists() else {}
    last = rows.groupby("code")["date"].max()
    need = [c for c, d in last.items()
            if c not in extra and (c not in cache or d >= cache[c].index.max())]
    print(f"가격: 캐시 {len(cache):,}종목 · 추가로 받을 종목 {len(need):,}")
    for k, code in enumerate(need):
        try:
            start = "2024-10-01" if code not in cache else "2026-08-01"
            extra[code] = fdr.DataReader(code, start)[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            extra[code] = None
        if k % 100 == 99:
            pickle.dump(extra, open(PX, "wb"))
            print(f"  {k + 1}/{len(need)}")
    pickle.dump(extra, open(PX, "wb"))
    out = {}
    for code in last.index:
        a, b = cache.get(code), extra.get(code)
        parts = [x[["Open", "High", "Low", "Close", "Volume"]] for x in (a, b) if x is not None and len(x)]
        if parts:
            d = pd.concat(parts)
            out[code] = d[~d.index.duplicated(keep="first")].sort_index()
    return out


def attach(rows: pd.DataFrame, px: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rec = []
    for r in rows.itertuples(index=False):
        d = px.get(r.code)
        if d is None or r.date not in d.index:
            continue
        i = d.index.get_loc(r.date)
        if i == 0 or i + 1 >= len(d):
            continue
        c0, c, o1 = d["Close"].iloc[i - 1], d["Close"].iloc[i], d["Open"].iloc[i + 1]
        v, v1 = d["Volume"].iloc[i], d["Volume"].iloc[i + 1]
        if not (c0 > 0 and c > 0 and o1 > 0 and v > 0 and v1 > 0):   # 거래정지 봉 제외
            continue
        ret = c / c0 - 1
        rec.append({**r._asdict(), "close": c, "day_ret": ret,
                    "limit_up": ret >= config.LIMIT_UP_PCT,
                    "tval": c * v,
                    "y": o1 / c - 1 - config.COST,
                    "next_date": d.index[i + 1]})
    return pd.DataFrame(rec)


if __name__ == "__main__":
    rows = load_rows()
    print(f"사유 있는 종목-일: {len(rows):,}건 · {rows['date'].nunique()}일")
    panel = attach(rows, load_prices(rows))
    panel.to_parquet(OUT, index=False)
    print(f"저장 {OUT.name}: {len(panel):,}건 · {panel['date'].nunique()}일 "
          f"· {panel['date'].min().date()}~{panel['date'].max().date()}")
    print(f"상한가 마감 비중 {panel['limit_up'].mean():.1%} · 출처 {panel['src'].value_counts().to_dict()}")
    # 등락률 대조: 원문에 적힌 등락률과 가격으로 계산한 값이 어긋나면 종목 매칭 오류다
    bad = (panel["chg"] / 100 - panel["day_ret"]).abs() > 0.02
    print(f"원문 등락률과 2%p 넘게 어긋나는 행 {bad.sum():,}건 ({bad.mean():.1%})")
