# -*- coding: utf-8 -*-
"""HollyKR OHLCV 캐시를 FinanceDataReader로 갱신해 로컬 프로젝트 캐시를 만든다.

원본(HollyKR `.cache/ohlcv/ohlcv_cache.pkl`)은 **읽기 전용**으로 두고,
갱신본을 이 프로젝트의 `data/ohlcv_cache.pkl` 에 따로 쓴다.

- 기존 종목: 마지막 날짜 다음날부터 증분 수집
- 미커버 종목: `--include-new` 지정 시 전체 기간 신규 수집

pickle 사용 주의: 읽는 pkl은 전부 **1st-party 로컬 캐시**다 — HollyKR 프로젝트가
자기 머신에서 쓴 것, 그리고 이 스크립트가 직접 쓴 것. 외부에서 받은 파일을 역직렬화하지
않는다. 원본 형식(dict[str, DataFrame])을 그대로 이어 쓰기 위해 pickle을 유지한다.

사용:
    python scripts/refresh_cache.py                 # 기존 종목만 증분
    python scripts/refresh_cache.py --include-new   # 미커버 종목도 신규 수집
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HOLLYKR_CACHE = Path(
    r"C:\Users\hsh\Desktop\vibecoding\주식 자동매매시스템과 관련  프로젝트\.cache"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "data" / "ohlcv_cache.pkl"
START_DATE = "2019-01-01"
COLS = ["Open", "High", "Low", "Close", "Volume", "Change"]


def load_source_cache() -> dict[str, pd.DataFrame]:
    """HollyKR 캐시에서 code별 가장 긴 시계열만 남긴 dict를 만든다."""
    path = HOLLYKR_CACHE / "ohlcv" / "ohlcv_cache.pkl"
    with open(path, "rb") as f:
        raw = pickle.load(f)
    best: dict[str, pd.DataFrame] = {}
    for key, df in raw.items():
        m = re.match(r"(\d+)_(\d+)$", key)
        if not m or df is None or len(df) == 0:
            continue
        code = m.group(1)
        if code not in best or len(df) > len(best[code]):
            best[code] = df
    return best


def load_existing() -> dict[str, pd.DataFrame]:
    """이 프로젝트가 이미 만든 갱신본이 있으면 그걸 이어서 쓴다."""
    if OUT_PATH.exists():
        with open(OUT_PATH, "rb") as f:
            return pickle.load(f)
    return load_source_cache()


def krx_codes() -> list[str]:
    """KRX 보통주 코드 목록.

    fdr.StockListing('KOSPI'/'KOSDAQ'/'KRX')는 KRX OTP 문제로 이 환경에서 실패한다
    (기존 웹사이트 프로젝트에서도 확인된 알려진 버그). 그래서:
      1) HollyKR 로컬 krx_master_*.pkl 로 보통주 판별(스팩·리츠·ETF 제외)
      2) fdr.StockListing('KRX-DESC') 로 신규 상장을 보완
    """
    import FinanceDataReader as fdr

    codes: set[str] = set()
    excluded: set[str] = set()

    # 1) 로컬 마스터 — is_common/is_spac/is_reit/is_etf 플래그가 있다
    masters = sorted(HOLLYKR_CACHE.glob("krx_master_*.pkl"))
    if masters:
        try:
            m = pickle.load(open(masters[-1], "rb"))
            keep = (m["is_common"] & ~m["is_spac"] & ~m["is_reit"] & ~m["is_etf"])
            codes |= {str(c).zfill(6) for c in m.loc[keep, "종목코드"]}
            excluded |= {str(c).zfill(6) for c in m.loc[~keep, "종목코드"]}
            print(f"  로컬 마스터({masters[-1].name}): 보통주 {len(codes)} "
                  f"· 제외 {len(excluded)}")
        except Exception as e:
            print(f"  [warn] 로컬 마스터 실패: {e}")

    # 2) KRX-DESC 로 신규 상장 보완 (마스터에서 이미 제외된 종목은 제외 유지)
    try:
        d = fdr.StockListing("KRX-DESC")
        live = {str(c).zfill(6) for c in d["Code"].dropna()}
        # 스팩은 종목명으로 한 번 더 거른다
        spac = {str(r.Code).zfill(6) for r in d.itertuples(index=False)
                if "스팩" in str(r.Name)}
        added = (live - excluded - spac) - codes
        codes |= added
        print(f"  KRX-DESC: 전체 {len(live)} · 신규 보완 {len(added)}")
    except Exception as e:
        print(f"  [warn] KRX-DESC 실패: {e}")

    return sorted(codes)


def fetch(code: str, start: str, end: str) -> pd.DataFrame | None:
    import FinanceDataReader as fdr

    try:
        df = fdr.DataReader(code, start, end)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    for c in COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[COLS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-new", action="store_true",
                    help="캐시에 없는 종목도 전체 기간 신규 수집")
    ap.add_argument("--limit", type=int, default=None,
                    help="처리 종목 수 제한 (테스트용)")
    args = ap.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache = load_existing()
    print(f"기존 캐시: {len(cache)}종목")
    src_max = max(df.index[-1] for df in cache.values())
    print(f"기존 최신일: {src_max.date()}")

    today = datetime.now().date()
    end = today.strftime("%Y-%m-%d")

    targets = list(cache.keys())
    new_codes: list[str] = []
    if args.include_new:
        all_codes = krx_codes()
        new_codes = [c for c in all_codes if c not in cache]
        print(f"KRX 전체 {len(all_codes)}종목 · 미커버 {len(new_codes)}종목 신규 수집 대상")

    if args.limit:
        targets = targets[: args.limit]
        new_codes = new_codes[: args.limit]

    t0 = time.time()
    updated = skipped = failed = 0

    # 1) 기존 종목 증분
    for i, code in enumerate(targets, 1):
        df = cache[code]
        last = df.index[-1].date()
        if last >= today - timedelta(days=1):
            skipped += 1
            continue
        start = (last + timedelta(days=1)).strftime("%Y-%m-%d")
        add = fetch(code, start, end)
        if add is None:
            failed += 1
        else:
            add = add[~add.index.isin(df.index)]
            if len(add):
                cache[code] = pd.concat([df, add]).sort_index()
                updated += 1
            else:
                skipped += 1
        if i % 200 == 0:
            print(f"  증분 {i}/{len(targets)} · 갱신 {updated} · 스킵 {skipped} "
                  f"· 실패 {failed} · {time.time()-t0:.0f}s", flush=True)

    print(f"증분 완료: 갱신 {updated} · 스킵 {skipped} · 실패 {failed} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # 2) 신규 종목
    added = 0
    for i, code in enumerate(new_codes, 1):
        df = fetch(code, START_DATE, end)
        if df is not None and len(df) >= 300:
            cache[code] = df
            added += 1
        if i % 100 == 0:
            print(f"  신규 {i}/{len(new_codes)} · 추가 {added} "
                  f"· {time.time()-t0:.0f}s", flush=True)
            with open(OUT_PATH, "wb") as f:   # 중간 저장 (중단 대비)
                pickle.dump(cache, f)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(cache, f)

    new_max = max(df.index[-1] for df in cache.values())
    print(f"\n저장: {OUT_PATH}")
    print(f"총 {len(cache)}종목 (신규 {added}) · 최신일 {new_max.date()} "
          f"· 총 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
