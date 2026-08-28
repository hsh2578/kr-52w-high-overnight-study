# -*- coding: utf-8 -*-
"""키움 REST API로 종목별 투자자 수급(개인·외국인·기관) 일별 이력을 수집한다.

배경: 한국 시장의 익일 수익률 예측에서 수급은 가장 유력한 후보 변수인데, 다른 경로는
전 기간 확보가 안 됐다 — KRX Open API에 투자자별 엔드포인트가 없고(404), KIS는 30일치만
주고(FHKST01010900), HollyKR 로컬 캐시는 3개월(2025-12~2026-03)뿐이다.
키움 ka10059는 요청일 기준 과거 100거래일을 주고 2019-11까지 도달하므로, 날짜를 옮겨가며
호출하면 전 기간을 덮을 수 있다(실측: 3.0 req/s, 10/10 성공).

**조회 전용이다.** 주문·계좌 관련 엔드포인트는 절대 호출하지 않는다.

병렬 처리: 종목 단위로 4개 워커. 실측 동시성 한도는 workers=2가 초당 4.5회(30/30 성공),
4가 초당 7.1회(30/30), 8이 초당 8.9회(24/30 = 20% 실패), 12가 초당 10.9회(26/30)였다.
성공률 100%를 유지하는 4워커를 택하고 실패 시 재시도한다(약 43분 예상).
한 종목 안에서는 다음 요청 날짜가 이전 응답에 의존하므로 순차 처리한다.

사용:
    python scripts/collect_investor.py            # events.parquet의 종목 전체
    python scripts/collect_investor.py --limit 5  # 테스트
"""
from __future__ import annotations

import argparse
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

MAX_WORKERS = 4
MAX_RETRY = 3

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = Path(r"C:\Users\hsh\Desktop\모의투자대회\.env")
EVENTS = PROJECT_ROOT / "data" / "events.parquet"
OUT = PROJECT_ROOT / "data" / "investor.pkl"

BASE = "https://api.kiwoom.com"
API_ID = "ka10059"                 # 종목별 투자자·기관별 요청 (조회 전용)
ROWS_PER_CALL = 100                # 1회 응답이 덮는 거래일 수
SLEEP = 0.30                       # 초당 약 3회 (실측 한도)

# 응답 필드 → 우리 컬럼명
FIELDS = {
    "dt": "date",
    "ind_invsr": "indiv",          # 개인
    "frgnr_invsr": "foreign",      # 외국인
    "orgn": "inst",                # 기관 합계
    "fnnc_invt": "fin_invt",       # 금융투자
    "insrnc": "insurance",         # 보험
    "acc_trde_prica": "trd_value",  # 누적 거래대금
}


def load_env() -> dict:
    env = {}
    for line in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def get_token(env: dict) -> str:
    ak = env.get("KIWOOM_APP_KEY", "")
    sk = env.get("KIWOOM_APP_SECRET", "") or env.get("KIWOOM_SECRET_KEY", "")
    r = requests.post(f"{BASE}/oauth2/token",
                      json={"grant_type": "client_credentials",
                            "appkey": ak, "secretkey": sk}, timeout=20)
    r.raise_for_status()
    tok = r.json().get("token")
    if not tok:
        raise RuntimeError("키움 토큰 발급 실패")
    return tok


def _to_num(x) -> float:
    """'+1,234' / '--567' 같은 표기를 float으로. 부호 중복은 하나로 접는다."""
    s = str(x).replace(",", "").strip()
    if not s or s in {"-", "+"}:
        return 0.0
    neg = s.startswith("-")
    s = s.lstrip("+-")
    try:
        v = float(s)
    except ValueError:
        return 0.0
    return -v if neg else v


def fetch_chunk(token: str, code: str, anchor: str) -> pd.DataFrame:
    """anchor(YYYYMMDD) 기준 과거 100거래일 수급. 실패 시 빈 프레임."""
    h = {"authorization": f"Bearer {token}", "api-id": API_ID,
         "Content-Type": "application/json;charset=UTF-8"}
    body = {"dt": anchor, "stk_cd": code, "amt_qty_tp": "1",
            "trde_tp": "0", "unit_tp": "1000"}
    try:
        r = requests.post(f"{BASE}/api/dostk/stkinfo", headers=h,
                          json=body, timeout=25)
        if r.status_code != 200:
            return pd.DataFrame()
        j = r.json()
        if j.get("return_code") != 0:
            return pd.DataFrame()
        rows = j.get("stk_invsr_orgn") or []
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()

    out = []
    for x in rows:
        rec = {}
        for src, dst in FIELDS.items():
            rec[dst] = x.get(src)
        out.append(rec)
    df = pd.DataFrame(out)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    for c in ("indiv", "foreign", "inst", "fin_invt", "insurance", "trd_value"):
        df[c] = df[c].map(_to_num)
    return df.dropna(subset=["date"]).sort_values("date")


def collect_code(token: str, code: str, start: pd.Timestamp,
                 end: pd.Timestamp) -> pd.DataFrame:
    """end에서 start까지 100일씩 거슬러 올라가며 모은다."""
    frames = []
    anchor = end
    guard = 0
    while anchor >= start and guard < 40:
        guard += 1
        df = fetch_chunk(token, code, anchor.strftime("%Y%m%d"))
        time.sleep(SLEEP)
        if df.empty:
            break
        frames.append(df)
        oldest = df["date"].min()
        if oldest <= start or oldest >= anchor:
            break
        anchor = oldest - timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates("date")
    return out[out["date"] >= start - timedelta(days=5)].sort_values("date")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-up", action="store_true",
                    help="이미 수집된 종목도 '끊긴 뒤 구간'만 이어서 채운다. "
                         "이 수집은 과거 방향으로만 진행돼 종목의 98.4%가 "
                         "2026-08 이전에 끊겨 있었다(종료일 중앙값 2025-12-16).")
    ap.add_argument("--events", default=None,
                    help="종목·기간의 출처 parquet. 기본은 data/events.parquet "
                         "(52주 신고가). 급등주 연구는 그 이벤트 파일을 넘겨야 "
                         "종목 집합과 기간이 맞는다.")
    args = ap.parse_args()

    ev = pd.read_parquet(args.events or EVENTS)
    span = ev.groupby("code")["date"].agg(["min", "max"])
    # 수급 피처는 이벤트 직전 20거래일을 보므로 시작을 40일 앞당긴다
    span["min"] = span["min"] - timedelta(days=45)
    codes = list(span.index)
    if args.limit:
        codes = codes[: args.limit]
    print(f"대상 종목 {len(codes):,} | 기간 {ev.date.min().date()} ~ {ev.date.max().date()}",
          flush=True)

    env = load_env()
    token = get_token(env)
    print("키움 토큰 발급 완료", flush=True)

    store: dict[str, pd.DataFrame] = {}
    if OUT.exists():                      # 중단 후 재개 지원
        store = pickle.load(open(OUT, "rb"))
        print(f"기존 수집분 {len(store):,}종목 이어서 진행", flush=True)

    if args.top_up:
        # 이미 있는 종목이라도 '마지막 수집일 < 필요한 마지막 날짜'면 다시 대상에
        # 넣고, 그 종목의 시작점을 마지막 수집일로 당겨 꼬리만 받는다.
        todo = []
        for c in codes:
            need_end = span.loc[c, "max"]
            cur = store.get(c)
            if cur is None or not len(cur):
                todo.append(c)
                continue
            have_end = pd.to_datetime(cur["date"]).max()
            if have_end < need_end - timedelta(days=3):
                span.loc[c, "min"] = have_end       # 겹치는 며칠은 병합 때 정리
                todo.append(c)
        print(f"top-up 대상 {len(todo):,}종목 (전체 {len(codes):,})", flush=True)
    else:
        todo = [c for c in codes if not (c in store and len(store[c]))]
    print(f"수집 대상 {len(todo):,}종목 · 워커 {MAX_WORKERS}개", flush=True)

    lock = threading.Lock()
    t0 = time.time()
    state = {"done": 0, "failed": 0, "n": 0}

    def work(code: str):
        """한 종목을 끝까지 수집. 빈 결과면 재시도."""
        for attempt in range(MAX_RETRY):
            df = collect_code(token, code, span.loc[code, "min"],
                              span.loc[code, "max"])
            if not df.empty:
                return code, df
            time.sleep(0.5 * (attempt + 1))   # 백오프 후 재시도
        return code, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(work, c) for c in todo]
        for fut in as_completed(futures):
            code, df = fut.result()
            if args.top_up and code in store and len(store[code]) and len(df):
                df = (pd.concat([store[code], df], ignore_index=True)
                        .drop_duplicates(subset=["date"], keep="last")
                        .sort_values("date")
                        .reset_index(drop=True))
            with lock:
                state["n"] += 1
                if df.empty:
                    state["failed"] += 1
                else:
                    store[code] = df
                    state["done"] += 1
                if state["n"] % 100 == 0:
                    el = time.time() - t0
                    rate = state["n"] / max(el, 1e-9)
                    left = (len(todo) - state["n"]) / max(rate, 1e-9) / 60
                    print(f"  {state['n']}/{len(todo)} · 수집 {state['done']} "
                          f"· 실패 {state['failed']} · {el/60:.1f}분 경과 "
                          f"· 잔여 약 {left:.0f}분", flush=True)
                    with open(OUT, "wb") as f:    # 중간 저장
                        pickle.dump(store, f)

    failed = state["failed"]
    with open(OUT, "wb") as f:
        pickle.dump(store, f)
    tot = sum(len(v) for v in store.values())
    print(f"\n저장: {OUT}")
    print(f"종목 {len(store):,} · 총 {tot:,}행 · 실패 {failed} "
          f"· {(time.time()-t0)/60:.1f}분")
    return 0


if __name__ == "__main__":
    sys.exit(main())
