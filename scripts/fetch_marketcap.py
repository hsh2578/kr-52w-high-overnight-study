# -*- coding: utf-8 -*-
"""KRX Open API로 **시점별 상장주식수**를 받아 `data/list_shares.parquet`에 저장한다.

왜 상장주식수인가 (시가총액을 직접 받지 않고):
  KRX 일별 시세는 MKTCAP을 주지만, 2020~2026 전 거래일을 받으려면
  약 1,600일 x 2개 시장 = 3,200회 호출이 필요하다. 반면 상장주식수는
  유상증자·분할 같은 자본변동이 있을 때만 바뀌므로 **월 1회 스냅샷**으로
  거의 모든 변화를 잡는다(160회). 시가총액은
      시총(t) = 상장주식수(t 이전 가장 가까운 스냅샷) x 종가(t)
  로 계산한다.

왜 시점 데이터여야 하는가:
  현재 시가총액을 과거 이벤트에 적용하면 **생존편향·미래정보**가 된다.
  2020년에 300억이던 회사가 지금 2조라고 해서 그때 대형주였던 것이 아니다.
  이 프로젝트는 다른 저장소에서 바로 그 결함(현재 시총 스냅샷으로 과거를
  필터링)을 지적했으므로, 같은 실수를 되풀이하지 않는다.

사용: python scripts/fetch_marketcap.py [--start 202001] [--end 202608]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = Path(r"C:\Users\hsh\Desktop\모의투자대회\.env")
OUT = PROJECT_ROOT / "data" / "list_shares.parquet"
BASE = "https://data-dbg.krx.co.kr/svc/apis"
PATHS = ("sto/stk_bydd_trd", "sto/ksq_bydd_trd")   # 코스피 · 코스닥
MAX_HOLIDAY_SKIP = 7        # 월초가 휴장이면 이만큼 다음 날짜를 시도한다


def load_key() -> str:
    for line in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
        if line.strip().startswith("KRX_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("KRX_API_KEY 없음")


def month_starts(start: str, end: str) -> list[date]:
    y, m = int(start[:4]), int(start[4:6])
    ey, em = int(end[:4]), int(end[4:6])
    out = []
    while (y, m) <= (ey, em):
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_one(key: str, path: str, dd: str) -> list[dict]:
    for attempt in range(3):
        try:
            r = requests.get(f"{BASE}/{path}", headers={"AUTH_KEY": key},
                             params={"basDd": dd}, timeout=40)
            if r.status_code == 200:
                return r.json().get("OutBlock_1") or []
            if r.status_code in (429, 503):      # 혼잡 — 물러섰다 재시도
                time.sleep(2 * (attempt + 1))
                continue
            return []
        except requests.RequestException:
            time.sleep(1 + attempt)
    return []


def fetch_month(key: str, d0: date) -> list[dict]:
    """월초 영업일 스냅샷. 휴장이면 다음 날짜로 며칠 밀어본다."""
    for k in range(MAX_HOLIDAY_SKIP):
        dd = (d0 + timedelta(days=k)).strftime("%Y%m%d")
        rows = []
        for p in PATHS:
            rows += fetch_one(key, p, dd)
        if rows:
            return [{"date": pd.Timestamp(dd),
                     "code": str(x.get("ISU_CD", "")).zfill(6),
                     "shares": float(str(x.get("LIST_SHRS", "0")).replace(",", "") or 0)}
                    for x in rows]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="202001")
    ap.add_argument("--end", default=None, help="기본: 이번 달")
    ap.add_argument("--workers", type=int, default=4,
                    help="병렬 수. 너무 높이면 429가 나므로 4 권장.")
    args = ap.parse_args()
    end = args.end or date.today().strftime("%Y%m")

    key = load_key()
    months = month_starts(args.start, end)
    print(f"{len(months)}개월 x {len(PATHS)}개 시장 스냅샷 수집 "
          f"({args.start} ~ {end}) · 병렬 {args.workers}")

    t0 = time.time()
    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for got in ex.map(lambda d: fetch_month(key, d), months):
            rows += got
            done += 1
            if done % 12 == 0 or done == len(months):
                print(f"  {done}/{len(months)}개월 · 누적 {len(rows):,}행 "
                      f"· {time.time()-t0:.0f}s", flush=True)

    if not rows:
        raise RuntimeError("한 건도 받지 못했다")

    df = (pd.DataFrame(rows)
            .query("shares > 0")
            .drop_duplicates(["code", "date"])
            .sort_values(["code", "date"])
            .reset_index(drop=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print(f"\n저장: {OUT}")
    print(f"  {len(df):,}행 · 종목 {df['code'].nunique():,}개 · "
          f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    miss = len(months) - df["date"].nunique()
    if miss:
        print(f"  [warn] 스냅샷 {miss}개월이 비었다 — 그 구간은 직전 값으로 이월된다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
