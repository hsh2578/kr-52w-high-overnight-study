# -*- coding: utf-8 -*-
"""도메인 지식 기반 데이터 검증.

"구조적으로 컬럼이 있다"와 "값이 실제로 맞다"는 다르다. 이 스크립트는 한국 주식시장의
제도적 사실을 기준으로 캐시 데이터가 진짜인지 확인하고, KRX 공식 API를 독립 출처로
교차검증한다.

검사 항목:
  A. 가격제한폭 ±30% 위반 (2015-06 이후 KRX 제도)
  B. OHLC 대소관계 위반 (Low <= Open,Close <= High)
  C. 거래량 0인데 가격이 움직인 봉
  D. 거래일이 주말인 봉
  E. 라벨(익일 시가 수익률) 이상치
  F. KRX 공식 등락률과 캐시 등락률 교차검증
  G. 재료 CSV의 당일등락률과 캐시 등락률 교차검증 (독립 출처 2종)

사용: python scripts/audit_data.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = PROJECT_ROOT / "data" / "ohlcv_cache.pkl"
CATALYST = Path(r"C:\Users\hsh\Desktop\모의투자대회\data\특징주_재료_상승률_2년.csv")
ENV = Path(r"C:\Users\hsh\Desktop\모의투자대회\.env")
KRX_BASE = "https://data-dbg.krx.co.kr/svc/apis"

LIMIT = 0.30          # 가격제한폭 ±30% (2015-06-15 이후)
TOL = 0.005           # 제한폭 판정 여유 (수정주가 반올림 오차)


def load_env() -> dict:
    env = {}
    for line in ENV.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def krx_daily(auth_key: str, date: str) -> pd.DataFrame:
    """KRX 공식 일별매매정보 (유가증권 + 코스닥)."""
    frames = []
    for path in ("/sto/stk_bydd_trd", "/sto/ksq_bydd_trd"):
        r = requests.get(KRX_BASE + path, headers={"AUTH_KEY": auth_key},
                         params={"basDd": date}, timeout=30)
        r.raise_for_status()
        rows = r.json().get("OutBlock_1") or []
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["code"] = df["ISU_CD"].astype(str).str.zfill(6)
    for c in ("TDD_CLSPRC", "FLUC_RT", "ACC_TRDVAL", "MKTCAP"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    return df


def main() -> int:
    cache = pickle.load(open(CACHE, "rb"))
    print(f"검증 대상: {len(cache):,}종목\n")

    # ---- A~D: 제도적 제약 위반 검사 -------------------------------------
    viol_limit = viol_ohlc = viol_volzero = viol_weekend = 0
    total_bars = 0
    limit_examples = []

    for code, df in cache.items():
        o = df["Open"].to_numpy(float)
        h = df["High"].to_numpy(float)
        l = df["Low"].to_numpy(float)
        c = df["Close"].to_numpy(float)
        v = df["Volume"].to_numpy(float)
        total_bars += len(c)

        # A. 가격제한폭: 정지봉(NaN)과 첫봉 제외
        with np.errstate(invalid="ignore", divide="ignore"):
            chg = np.full(len(c), np.nan)
            chg[1:] = c[1:] / c[:-1] - 1.0
        bad = np.abs(chg) > LIMIT + TOL
        bad &= np.isfinite(chg)
        nb = int(bad.sum())
        if nb:
            viol_limit += nb
            if len(limit_examples) < 5:
                i = int(np.argmax(np.abs(np.nan_to_num(chg))))
                limit_examples.append((code, str(df.index[i].date()),
                                       round(float(chg[i]) * 100, 1)))

        # B. OHLC 대소관계 (정지봉은 Open이 NaN이라 자동 제외)
        ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
        viol_ohlc += int((ok & ((l > o) | (l > c) | (h < o) | (h < c) | (h < l))).sum())

        # C. 거래량 0인데 가격 변동
        volzero = np.isfinite(v) & (v == 0)
        moved = np.zeros(len(c), bool)
        moved[1:] = c[1:] != c[:-1]
        viol_volzero += int((volzero & moved).sum())

        # D. 주말 봉
        viol_weekend += int((df.index.dayofweek >= 5).sum())

    print("=== A~D. 제도적 제약 검사 ===")
    print(f"총 봉 수: {total_bars:,}")
    print(f"A. 가격제한폭 ±30% 위반 : {viol_limit:,} ({viol_limit/total_bars*100:.4f}%)")
    if limit_examples:
        print(f"   예시(종목,날짜,등락률): {limit_examples}")
        print("   → 신규상장 첫날·액면분할·병합은 제한폭 예외라 일부는 정상이다.")
    print(f"B. OHLC 대소관계 위반    : {viol_ohlc:,}")
    print(f"C. 거래량 0인데 가격변동 : {viol_volzero:,}")
    print(f"D. 주말 거래일          : {viol_weekend:,}")

    # ---- F: KRX 공식 등락률과 교차검증 ----------------------------------
    print("\n=== F. KRX 공식 API 교차검증 ===")
    env = load_env()
    auth = env.get("KRX_API_KEY", "")
    checked = 0
    for date in ("20260827", "20260814", "20260703"):
        try:
            k = krx_daily(auth, date)
        except Exception as e:
            print(f"  {date}: KRX 호출 실패 {type(e).__name__}")
            continue
        if k.empty:
            print(f"  {date}: KRX 응답 없음(휴장일 가능)")
            continue
        ts = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:]}")
        rows = []
        for _, r in k.iterrows():
            df = cache.get(r["code"])
            if df is None or ts not in df.index:
                continue
            i = df.index.get_loc(ts)
            if i < 1:
                continue
            cc = float(df["Close"].iloc[i]); pc = float(df["Close"].iloc[i - 1])
            if not (np.isfinite(cc) and np.isfinite(pc) and pc > 0):
                continue
            rows.append({"code": r["code"],
                         "krx": float(r["FLUC_RT"]),
                         "cache": (cc / pc - 1.0) * 100})
        m = pd.DataFrame(rows)
        if m.empty:
            print(f"  {date}: 매칭 종목 없음")
            continue
        d = (m["krx"] - m["cache"]).abs()
        checked += len(m)
        print(f"  {date}: 매칭 {len(m):,}종목 | 등락률 차이 "
              f"중앙 {d.median():.4f}%p, 평균 {d.mean():.4f}%p, "
              f"0.1%p 초과 {int((d > 0.1).sum())}건 ({(d > 0.1).mean()*100:.2f}%)")
        if (d > 0.1).any():
            worst = m.loc[d.nlargest(3).index]
            print(f"     최대 불일치: "
                  f"{[(r.code, round(r.krx,2), round(r.cache,2)) for r in worst.itertuples()]}")

    # ---- G: 재료 CSV와 교차검증 (완전히 독립적인 제3의 출처) --------------
    print("\n=== G. 재료 CSV 당일등락률 교차검증 (OCR 출처, 독립) ===")
    cs = pd.read_csv(CATALYST)
    cs.columns = ["date", "name", "code", "chg_s", "chg", "ctype",
                  "reason", "d1", "d2", "d3", "cum3"]
    cs["code"] = cs["code"].astype(str).str.zfill(6)
    cs["date"] = pd.to_datetime(cs["date"])
    rows = []
    for r in cs.itertuples(index=False):
        df = cache.get(r.code)
        if df is None or r.date not in df.index:
            continue
        i = df.index.get_loc(r.date)
        if i < 1:
            continue
        cc = float(df["Close"].iloc[i]); pc = float(df["Close"].iloc[i - 1])
        if not (np.isfinite(cc) and np.isfinite(pc) and pc > 0):
            continue
        rows.append({"csv": float(r.chg), "cache": (cc / pc - 1.0) * 100})
    m = pd.DataFrame(rows)
    if len(m):
        d = (m["csv"] - m["cache"]).abs()
        print(f"  매칭 {len(m):,}건 | 차이 중앙 {d.median():.4f}%p, "
              f"평균 {d.mean():.4f}%p")
        print(f"  1%p 초과 불일치: {int((d > 1).sum())}건 ({(d > 1).mean()*100:.2f}%)")
        print("  → 재료 CSV는 OCR 전사물이라 일부 오차는 예상된다. "
              "대량 불일치면 날짜 정렬이 어긋난 것이다.")
    else:
        print("  매칭 없음")

    # ---- E: 라벨 이상치 --------------------------------------------------
    print("\n=== E. 라벨(익일 시가 수익률) 분포 온전성 ===")
    ev = PROJECT_ROOT / "data" / "events.parquet"
    if ev.exists():
        e = pd.read_parquet(ev)
        y = e["y_raw"].dropna()
        print(f"  이벤트 {len(e):,} | y_raw 평균 {y.mean()*100:+.4f}% "
              f"중앙 {y.median()*100:+.4f}% 표준편차 {y.std()*100:.2f}%")
        print(f"  |y_raw| > 30% : {int((y.abs() > 0.30).sum())}건  "
              f"(가격제한폭상 있으면 안 된다)")
        print(f"  최소 {y.min()*100:+.2f}% / 최대 {y.max()*100:+.2f}%")
    else:
        print("  data/events.parquet 없음 — Task 3 실행 후 재확인")
    return 0


if __name__ == "__main__":
    sys.exit(main())
