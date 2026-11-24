# -*- coding: utf-8 -*-
"""텔레그램 급등 사유 글 → 종목·일자 표.

두 채널만 파싱한다(형식이 고정돼 있어 정규식으로 충분하다):
  AWAKE - 마켓 브리핑
    주요 상승 종목 현황        : 10:01(장중) · 15:33(마감 직후), 종목명·시총·등락률·사유
    주도 섹터/테마 현황        : 09 · 11 · 15시, 테마별 종목·등락률·거래대금·신고가 표시
  타점 읽어주는 여자(타자)
    상한가 및 급등주           : 18~19시, 그날의 상한가·급등주 전체와 사유(종목코드 포함)
    시간외/애프터마켓 특이종목 : 18~19시
    (AWAKE의 주요 상승 종목 현황을 전달한 글도 여기 올라온다 — 같이 읽는다)

핵심은 `ts`(게시 시각)다. 종가 매수 전략에 쓸 수 있는 것은 ts < 15:20 인 행뿐이다.
AWAKE는 전송 시각이 비어 있는 글이 있어 제목의 "(YYYY-MM-DD HH:MM 기준)"을 쓴다.

DB는 읽기 전용으로 연다.
출력: data/tg_reasons.parquet (사유) · data/tg_themes.parquet (테마 구성)
"""
from __future__ import annotations

import pickle
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surge_d1 import config  # noqa: E402

TG_DB = config.external("telegram.db")
OUT = config.PROJECT_ROOT / "data" / "tg_reasons.parquet"
THEMES_OUT = config.PROJECT_ROOT / "data" / "tg_themes.parquet"

AWAKE = "AWAKE - 마켓 브리핑"
TAJA = "타점 읽어주는 여자(타자)"

_AW_HEAD = re.compile(r"주요 상승 종목 현황\((\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) 기준\)")
_AW_ITEM = re.compile(r"^\*\*(.+?)\*\* \(([\d.,조억 ]+)\) ([+-][\d.]+)%\s*\n((?:(?!\*\*).*\n?)*)", re.M)
# 종목명 (코드) (등락률) 이 한 줄에 있거나 세 줄에 나뉘어 있다 — 둘 다 받는다
_TJ_ITEM = re.compile(r"^([^\n()]+?)[ \t]*\n?\(([0-9A-Z]{6})\)[ \t]*\n?"
                      r"\(\s*([+-]?[\d.]+)%\s*\)[ \t]*\n(.+)", re.M)
_TH_HEAD = re.compile(r"^\*\*✅ (.+?)\*\*\s*$", re.M)
_TH_ITEM = re.compile(r"^\[\s*([+-]?[\d.]+)%/\s*([\d.,조억 ]+)\]\s*([^/\n]+?)(?:/\s*(.+))?$", re.M)


def parse_awake(text: str) -> list[dict]:
    m = _AW_HEAD.search(text)
    if not m:
        return []
    ts = pd.Timestamp(f"{m.group(1)} {m.group(2)}")
    out = []
    for name, _cap, chg, body in _AW_ITEM.findall(text):
        reason = " ".join(x.strip(" -") for x in body.strip().splitlines() if x.strip())
        out.append({"ts": ts, "name": name.strip(), "code": None,
                    "chg": float(chg), "reason": reason})
    return out


def parse_taja(text: str, ts) -> list[dict]:
    if ts is None:
        return []
    t = pd.Timestamp(ts).tz_localize(None)
    return [{"ts": t, "name": n.strip(), "code": c, "chg": float(g), "reason": r.strip()}
            for n, c, g, r in _TJ_ITEM.findall(text)]


def parse_themes(text: str, ts) -> list[dict]:
    """'주도 섹터/테마 현황' → (테마, 종목, 등락률, 표시). 표시는 '52주 신고가' 등."""
    if ts is None or "주도 섹터/테마 현황" not in text:
        return []
    t = pd.Timestamp(ts).tz_localize(None)
    out, heads = [], list(_TH_HEAD.finditer(text))
    for k, h in enumerate(heads):
        block = text[h.end():heads[k + 1].start() if k + 1 < len(heads) else len(text)]
        for chg, _val, name, flag in _TH_ITEM.findall(block):
            out.append({"ts": t, "theme": h.group(1).strip(), "name": name.strip(),
                        "chg": float(chg), "flag": (flag or "").strip()})
    return out


def name_to_code() -> dict[str, str]:
    """종목명 → 코드. HollyKR의 1st-party 로컬 마스터 캐시(외부 파일 아님)."""
    masters = sorted(config.HOLLYKR_CACHE.glob("krx_master_*.pkl"))
    with open(masters[-1], "rb") as f:
        m = pickle.load(f)
    return dict(zip(m["종목명"].astype(str).str.strip(),
                    m["종목코드"].astype(str).str.zfill(6)))


def _con():
    return sqlite3.connect(f"file:{TG_DB.as_posix()}?mode=ro", uri=True)


def build() -> pd.DataFrame:
    con = _con()
    rows = []
    for (text,) in con.execute(
            "select text from messages where channel_name in (?, ?) and text like ?",
            (AWAKE, TAJA, "%주요 상승 종목 현황%")):
        rows += [{**r, "channel": "awake", "kind": "top_gainers"} for r in parse_awake(text)]
    for ts, text in con.execute(
            "select timestamp, text from messages where channel_name=? "
            "and (text like ? or text like ?)",
            (TAJA, "%특이종목%", "%상한가 및 급등주%")):
        kind = "surge_list" if "상한가 및 급등주" in text[:60] else "after_hours_movers"
        rows += [{**r, "channel": "taja", "kind": kind} for r in parse_taja(text, ts)]
    df = pd.DataFrame(rows)
    # 타점 글에는 코드가 있다 — 그 이름·코드 쌍으로 마스터(5월 기준)에 없는 신규 상장을 메운다
    n2c = {**name_to_code(),
           **dict(df.dropna(subset=["code"])[["name", "code"]].drop_duplicates("name").values)}
    df["code"] = df["code"].fillna(df["name"].map(n2c))
    # 자정을 넘겨 올라온 글(00~07시)은 전날 장에 대한 정리다
    late = df["ts"].dt.hour < 8
    df["date"] = (df["ts"] - pd.to_timedelta(late.astype(int), unit="D")).dt.normalize()
    df["slot"] = pd.cut(df["ts"].dt.hour * 60 + df["ts"].dt.minute,
                        [-1, 15 * 60 + 19, 15 * 60 + 59, 24 * 60],
                        labels=["intraday", "post_close", "after_hours"]).astype(str)
    df.loc[late, "slot"] = "after_hours"
    df["hm"] = df["ts"].dt.strftime("%H:%M")
    return (df.drop_duplicates(["kind", "ts", "name"])
              .sort_values(["ts", "kind"]).reset_index(drop=True))


def build_themes(n2c: dict[str, str]) -> pd.DataFrame:
    rows = []
    for ts, text in _con().execute(
            "select timestamp, text from messages where channel_name=? and text like ?",
            (AWAKE, "%주도 섹터/테마 현황%")):
        rows += parse_themes(text, ts)
    df = pd.DataFrame(rows)
    df["code"] = df["name"].map(n2c)
    df["date"] = df["ts"].dt.normalize()
    return df.drop_duplicates(["ts", "theme", "name"]).sort_values("ts").reset_index(drop=True)


def demo() -> None:
    a = parse_awake("**✅ 주요 상승 종목 현황(2026-03-12 10:00 기준)**\n\n"
                    "**에치에프알** (3,673억) +29.9%\n- 사유 하나\n- 사유 둘\n\n"
                    "**대한광통신** (1.1조) +16.3%\n사유 셋\n")
    assert [r["name"] for r in a] == ["에치에프알", "대한광통신"], a
    assert a[0]["reason"] == "사유 하나 사유 둘" and a[1]["chg"] == 16.3
    t = parse_taja("제목\n\n에이스토리\n(241840)\n( +30.48% )\n한한령 기대감\n\n"
                   "피스피스스튜디오\n(0117P0)\n( +18.71% )\n패션 테마\n",
                   "2026-11-18T19:38:02+09:00")
    assert [r["code"] for r in t] == ["241840", "0117P0"], t
    one = parse_taja("☑️** 3일 화요일 상한가 및 급등주**\n\n\nAPS (054620)\n(+30.00%)  \n수전해 진출\n\n"
                     "신도기연 (290520) ( +9.95% )  \n실적 발표\n", "2026-03-03T18:10:00+09:00")
    assert [(r["name"], r["code"], r["chg"]) for r in one] == [
        ("APS", "054620", 30.0), ("신도기연", "290520", 9.95)], one
    th = parse_themes("**📌 주도 섹터/테마 현황**\n\n**✅ 방산**\n\n[상승률/거래대금]\n"
                      "[+30.0%/ 1,110억] 빅텍/ 52주 신고가\n[+20.2%/ 1,088억] 삼양컴텍\n............\n\n"
                      "**✅ 금융/지주/밸류업**\n[+14.8%/ 3,766억] HMM\n", "2026-03-03T11:00:00+09:00")
    assert [(r["theme"], r["name"], r["flag"]) for r in th] == [
        ("방산", "빅텍", "52주 신고가"), ("방산", "삼양컴텍", ""), ("금융/지주/밸류업", "HMM", "")], th


if __name__ == "__main__":
    demo()
    df = build()
    df.to_parquet(OUT, index=False)
    print(f"저장 {OUT.name}: {len(df):,}행 · {df['date'].nunique()}일 "
          f"· {df['date'].min().date()}~{df['date'].max().date()}")
    print(df.groupby(["kind", "slot"]).agg(rows=("name", "size"),
                                          days=("date", "nunique"),
                                          no_code=("code", lambda s: s.isna().sum())))
    n2c = {**name_to_code(), **dict(df.dropna(subset=["code"])[["name", "code"]].values)}
    th = build_themes(n2c)
    th.to_parquet(THEMES_OUT, index=False)
    print(f"저장 {THEMES_OUT.name}: {len(th):,}행 · {th['date'].nunique()}일 "
          f"· 테마 {th['theme'].nunique()}개 · 신고가 표시 {th['flag'].str.contains('신고가').sum():,}행")
