# -*- coding: utf-8 -*-
"""KRX Open API로 유니버스를 확정해 `data/universe.json` 에 저장한다.

왜 필요한가: 이벤트의 16.3%(4,631건 / 260종목)가 ETF·ETN·ELW였다. 지수 추종 ETP는
상승장에서 52주 신고가를 계속 갱신하고, 재료가 존재하지 않으며, 갭이 개별 기업이 아니라
기초지수(해외 ETF면 미국 시장)에 좌우된다. 개별주 연구에 섞이면 결과가 오염된다.
추가로 리츠·외국주권·DR·인프라펀드 117건, 우선주도 제외 대상이다.

기존 HollyKR `krx_master_*.pkl` 의 `is_etf` 플래그는 신뢰할 수 없다 — 삼성전자(005930)가
`is_etf=True` 로 잘못 표시돼 있음을 실측 확인했다. 그래서 KRX 공식 API를 쓴다.

출력 `data/universe.json`:
    {"as_of": "YYYYMMDD",
     "common": [...],      # 보통주 코드 (연구 대상)
     "excluded": [...],    # 우선주·리츠·외국주권·DR·인프라·ETF·ETN·ELW
     "counts": {...}}

마스터에 없는 코드(상장폐지 등)는 어느 쪽에도 넣지 않는다 — 상장폐지 종목을 남기는 것이
생존편향을 줄이므로 `events` 쪽에서 "제외 목록에 없으면 통과" 규칙으로 살린다.

사용: python scripts/fetch_universe.py [--date YYYYMMDD]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surge_d1 import config  # noqa: E402
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = config.external(".env")
OUT = PROJECT_ROOT / "data" / "universe.json"
BASE = "https://data-dbg.krx.co.kr/svc/apis"

# 주식 기본정보 (보통주/우선주 구분용)
STOCK_PATHS = ("/sto/stk_isu_base_info", "/sto/ksq_isu_base_info")
# ETP — 전량 제외
ETP_PATHS = ("/etp/etf_bydd_trd", "/etp/etn_bydd_trd", "/etp/elw_bydd_trd")

# SECUGRP_NM 중 연구 대상으로 남길 값. 나머지(부동산투자회사·외국주권·주식예탁증권·
# 사회간접자본투융자회사·투자회사)는 개별 기업의 주식이 아니므로 제외한다.
KEEP_SECUGRP = {"주권"}


def load_key() -> str:
    for line in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("KRX_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("KRX_API_KEY 없음")


def fetch(key: str, path: str, date: str) -> list[dict]:
    r = requests.get(BASE + path, headers={"AUTH_KEY": key},
                     params={"basDd": date}, timeout=40)
    if r.status_code != 200:
        print(f"  [warn] {path} HTTP {r.status_code}")
        return []
    return r.json().get("OutBlock_1") or []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="기준일 YYYYMMDD (기본: 최근 영업일 탐색)")
    args = ap.parse_args()

    key = load_key()

    # 휴장일이면 빈 응답이 오므로 최대 10일 거슬러 올라가며 찾는다
    dates = ([args.date] if args.date
             else [(datetime.now() - timedelta(days=d)).strftime("%Y%m%d")
                   for d in range(0, 10)])

    stocks: list[dict] = []
    used = None
    for d in dates:
        got: list[dict] = []
        for p in STOCK_PATHS:
            got += fetch(key, p, d)
        if got:
            stocks, used = got, d
            break
    if not stocks:
        raise RuntimeError("주식 기본정보를 받지 못했다")
    print(f"기준일 {used} · 주식 기본정보 {len(stocks)}종목")

    common, excluded = set(), set()
    secugrp: dict[str, int] = {}
    stkcert: dict[str, int] = {}
    for x in stocks:
        code = str(x.get("ISU_SRT_CD", "")).zfill(6)
        sg = str(x.get("SECUGRP_NM", "")).strip()
        kind = str(x.get("KIND_STKCERT_TP_NM", "")).strip()
        secugrp[sg] = secugrp.get(sg, 0) + 1
        stkcert[kind] = stkcert.get(kind, 0) + 1
        if sg in KEEP_SECUGRP and kind == "보통주":
            common.add(code)
        else:
            excluded.add(code)

    print("  SECUGRP_NM:", dict(sorted(secugrp.items(), key=lambda x: -x[1])))
    print("  KIND_STKCERT_TP_NM:", dict(sorted(stkcert.items(), key=lambda x: -x[1])))

    etp_counts = {}
    for p in ETP_PATHS:
        rows = fetch(key, p, used)
        codes = {str(x.get("ISU_CD", "")).zfill(6) for x in rows}
        codes |= {str(x.get("ISU_SRT_CD", "")).zfill(6) for x in rows
                  if x.get("ISU_SRT_CD")}
        etp_counts[p.rsplit("/", 1)[-1]] = len(codes)
        excluded |= codes
    print("  ETP:", etp_counts)

    common -= excluded          # 충돌 시 제외가 우선
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "as_of": used,
        "common": sorted(common),
        "excluded": sorted(excluded),
        "counts": {"common": len(common), "excluded": len(excluded),
                   "secugrp": secugrp, "stkcert": stkcert, "etp": etp_counts},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {OUT}")
    print(f"보통주 {len(common):,} · 제외 {len(excluded):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
