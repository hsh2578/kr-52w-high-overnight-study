# -*- coding: utf-8 -*-
"""종가·시가 단일가 호가 스냅샷 적재 — 체결 가능성을 실측하기 위한 원자료.

왜 필요한가: 이 연구는 "종가에 사서 익일 시가에 판다"를 일봉으로만 검증했다.
상한가 종목은 종가 단일가에서 매수 잔량만 쌓여 못 사고, 그 밖의 종목도 내
주문이 종가를 밀어올린다. 일봉으로는 둘 다 잴 수 없다. 그리고 **호가·예상체결
자료는 과거로 소급 조회가 안 된다** — 오늘부터 쌓는 것 말고는 방법이 없다.

하는 일: 당일 상승률 상위 종목의 호가 10단계 잔량과 예상체결가·예상체결량을
그 시각 그대로 저장한다. 하루 네 번 부르면 된다.

    15:19  종가 단일가 직전 (결정 시점의 호가)
    15:29  종가 단일가 마감 직전 (예상 종가·예상 체결량·상한가 매수 잔량)
    15:31  종가 확정 직후 (남은 매수 잔량 = 못 산 물량)
    08:55  익일 시가 단일가 마감 직전 (전날 종목의 예상 시가)

08:55 호출은 **전날 파일에 있던 종목**을 다시 찍는다(오늘 상승률 상위가 아니라).

시세 조회만 한다. 주문 기능은 없다. 한투 클라이언트는 기존 것을 그대로 쓴다.
출력: data/snap/YYYYMMDD.jsonl (한 줄 = 종목 하나의 한 시각 스냅샷)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surge_d1 import config  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "snap"
sys.path.insert(0, str(config.external("kis_client_dir")))   # kis.py(시세 조회 클라이언트)가 있는 폴더

MIN_RISE = 5.0     # 당일 등락률 하한(%). 52주 신고가 돌파일은 대부분 이 위에 있다


def candidates(kis) -> list[str]:
    # ponytail: 상승률 상위는 시장당 최대 30건이라 하루 60종목이 상한이다.
    # 돌파 후보를 빠짐없이 잡으려면 장중 일봉으로 is_breakout을 직접 돌려야 한다.
    seen: dict[str, None] = {}
    for mkt in ("0001", "1001"):
        for x in kis.rank_rise(mkt, lo=MIN_RISE):
            seen.setdefault(x["stck_shrn_iscd"], None)
    return list(seen)


def prev_codes(today: str) -> list[str]:
    files = sorted(f for f in SNAP.glob("*.jsonl") if f.stem < today)
    if not files:
        return []
    with open(files[-1], encoding="utf-8") as f:
        return list(dict.fromkeys(json.loads(line)["code"] for line in f))


def snapshot(kis, code: str) -> dict:
    j = kis.get("/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
                "FHKST01010200", {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code})
    return {"book": j.get("output1") or {}, "exp": j.get("output2") or {}}


def main() -> int:
    import kis  # 기존 한투 시세 클라이언트(주문 없음)

    now = datetime.now()
    today = now.strftime("%Y%m%d")
    if now.weekday() >= 5 or kis.is_holiday(today):
        print("휴장일 — 건너뜀")
        return 0
    morning = now.strftime("%H%M") < "0900"
    codes = prev_codes(today) if morning else candidates(kis)
    if not codes:
        print("대상 종목 없음")
        return 0
    SNAP.mkdir(parents=True, exist_ok=True)
    out = SNAP / f"{today}.jsonl"
    n = 0
    with open(out, "a", encoding="utf-8") as f:
        for code in codes:
            s = snapshot(kis, code)
            if not s["book"]:
                continue
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                                "phase": "open" if morning else "close",
                                "code": code, **s}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{now:%H:%M:%S} {out.name}: {n}/{len(codes)}종목 저장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
