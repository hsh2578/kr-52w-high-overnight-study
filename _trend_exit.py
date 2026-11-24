# -*- coding: utf-8 -*-
"""추세추종 청산 — 돌파 전략에 맞는 규칙이면 AI 신호가 살아나는가.

앞선 검증에서 **고정 목표가 청산**(+10%/+15% 닿으면 매도)으로는 AI 우위가
사라졌다. 원인은 AI가 고르는 종목의 변동성이 양방향이라는 것이었다 —
최고 +20.6%를 얻으려면 최대 -13.9%를 견뎌야 하는데, 고정 손절이 그 전에
걸린다.

그런데 고정 목표가는 애초에 돌파 전략에 맞는 규칙이 아니다. 돌파는
**추세를 타는** 전략이므로, 이기는 것은 끝까지 들고 가고 **돌파가
실패했을 때만** 끊어야 한다. 그래서 네 가지 추세추종 청산을 잰다:

  MA3   : 종가가 3일 이동평균 아래로 내려가면 청산
  MA5   : 5일 이동평균 기준 (조금 더 느슨하게)
  돌파선 : 종가가 52주 고가 아래로 되돌아가면 = 돌파 실패로 보고 청산
  추격손절: 보유 중 최고 종가 대비 -10%

최대 보유는 40봉으로 늘린다 — 추세추종은 이기는 것을 오래 들고 가야
의미가 있으므로 20봉으로 자르면 규칙 자체를 무력화한다.

출력: outputs/trend_exit.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\hsh\Desktop\03_투자·코딩 프로젝트\급등주 분석프로그램")

from lightgbm import LGBMClassifier, LGBMRegressor      # noqa: E402
from scipy.stats import ttest_ind                       # noqa: E402

from surge_d1 import sources                            # noqa: E402

CSV = (r"C:\Users\hsh\Desktop\03_투자·코딩 프로젝트\vibecoding\주식웹사이트\국내주식웹사이트"
       r"\stock-screener-kr\ml\data\training_data_v2.csv")
OUT = r"C:\Users\hsh\Desktop\03_투자·코딩 프로젝트\급등주 분석프로그램\outputs\trend_exit.csv"

FEATS = ["breakout_pct", "volume_surge", "close_strength", "base_length",
         "volatility_contraction", "volume_dry_up", "ma200_slope",
         "pct_above_52w_low", "rs_vs_market", "market_return",
         "ma20_deviation", "liquidity", "days_since_ath", "atr_ratio"]
REG = dict(n_estimators=200, max_depth=3, learning_rate=0.01, subsample=0.8,
           reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1, n_jobs=-1)
CLF = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1)
COST = 0.21
MAXHOLD = 40


def run_exit(df: pd.DataFrame, i: int, rule: str, level: float) -> tuple:
    """봉 i 종가 매수 후 규칙대로 청산. (수익률%, 보유봉수)를 돌려준다.

    `level`은 그 이벤트의 52주 고가. 돌파선 규칙에서만 쓴다.
    모든 판정은 **종가**로 한다 — 장중 값으로 판정하면 그 가격에 실제로
    체결했다는 보장이 없어 결과가 낙관적으로 부풀려진다.
    """
    c = df["Close"].to_numpy(dtype=float)
    c0 = c[i]
    if not np.isfinite(c0) or c0 <= 0:
        return np.nan, 0
    end = min(i + MAXHOLD, len(df) - 1)
    peak = c0
    for j in range(i + 1, end + 1):
        p = c[j]
        if not np.isfinite(p) or p <= 0:
            continue
        peak = max(peak, p)
        hit = False
        if rule == "ma3":
            w = c[max(0, j - 2):j + 1]
            hit = np.isfinite(w).all() and p < w.mean()
        elif rule == "ma5":
            w = c[max(0, j - 4):j + 1]
            hit = np.isfinite(w).all() and p < w.mean()
        elif rule == "breakout":          # 52주 고가 아래로 되돌아가면 실패
            hit = np.isfinite(level) and level > 0 and p < level
        elif rule == "trail10":
            hit = p <= peak * 0.90
        if hit:
            return (p / c0 - 1) * 100 - COST, j - i
    p = c[end]
    if not np.isfinite(p) or p <= 0:
        return np.nan, end - i
    return (p / c0 - 1) * 100 - COST, end - i


def main() -> int:
    d = pd.read_csv(CSV)
    d["breakout_date"] = pd.to_datetime(d["breakout_date"])
    d["ticker"] = d["ticker"].astype(str).str.zfill(6)
    d = d.dropna(subset=FEATS + ["max_gain_20d", "gain_at_day_20", "high_52w"])
    d = d.sort_values("breakout_date").reset_index(drop=True)
    d["hit"] = (d["max_gain_20d"] >= 15).astype(int)

    cut = d["breakout_date"].quantile(0.70)
    tr, te = d[d["breakout_date"] <= cut], d[d["breakout_date"] > cut].copy()
    print(f"학습 {len(tr):,}건 → 검정 {len(te):,}건 "
          f"({te.breakout_date.min().date()} ~ {te.breakout_date.max().date()})\n")

    clf = LGBMClassifier(**CLF).fit(tr[FEATS].to_numpy(float), tr["hit"].to_numpy(int))
    reg = LGBMRegressor(**REG).fit(tr[FEATS].to_numpy(float),
                                   tr["max_gain_20d"].to_numpy(float))
    X = te[FEATS].to_numpy(float)
    te["ai_score"] = (clf.predict_proba(X)[:, 1] * 100 * 0.7
                      + np.clip(reg.predict(X) * 3, 0, 100) * 0.3)

    ohlcv = sources.load_ohlcv()
    RULES = ["ma3", "ma5", "breakout", "trail10"]
    for r_ in RULES:
        te[r_] = np.nan
        te[r_ + "_hold"] = np.nan
    miss = 0
    for idx, row in te.iterrows():
        df = ohlcv.get(row["ticker"])
        if df is None:
            miss += 1
            continue
        pos = df.index.get_indexer([row["breakout_date"]])[0]
        if pos < 0 or pos + 1 >= len(df):
            miss += 1
            continue
        for r_ in RULES:
            v, h = run_exit(df, pos, r_, float(row["high_52w"]))
            te.at[idx, r_] = v
            te.at[idx, r_ + "_hold"] = h
    print(f"경로 시뮬레이션 {len(te)-miss:,}건 매칭 (실패 {miss})\n")

    def block(x, name, base=None):
        row = {"구분": name, "픽": len(x)}
        for r_ in RULES:
            row[r_] = x[r_].mean()
            row[r_ + "_win"] = (x[r_] > 0).mean() * 100
            row[r_ + "_hold"] = x[r_ + "_hold"].mean()
        if base is not None and len(x) > 1:
            t = ttest_ind(x["breakout"].dropna(), base["breakout"].dropna(),
                          equal_var=False)
            row["t"] = t.statistic
            row["p"] = t.pvalue
        return row

    rows = [block(te, "① 그냥 돌파 (전부)")]
    for q, lab in ((0.80, "② AI 상위 20%"), (0.90, "③ AI 상위 10%"),
                   (0.95, "④ AI 상위 5%")):
        rows.append(block(te[te["ai_score"] >= te["ai_score"].quantile(q)],
                          lab, base=te))
    res = pd.DataFrame(rows)

    print(f"{'구분':<22}{'픽':>6}" + "".join(f"{r:>18}" for r in RULES))
    for _, r in res.iterrows():
        line = f"{r['구분']:<22}{int(r['픽']):>6}"
        for r_ in RULES:
            line += f"{r[r_]:>9.2f}%({r[r_+'_win']:>4.1f}%)"
        print(line)
    print(f"\n{'':<28}" + "".join(f"{'평균보유 '+f'{res.iloc[0][r+chr(95)+chr(104)+'old']:.1f}봉':>18}" for r in RULES))

    print("\n돌파선 청산 기준 유의성 (기저 대비):")
    for _, r in res.iterrows():
        if pd.notna(r.get("t")):
            print(f"  {r['구분']:<22} t={r['t']:+.2f}  p={r['p']:.3f}")

    res.to_csv(OUT, index=False)
    print(f"\n저장: {OUT}")
    print("(괄호 안은 승률. 왕복비용 0.21% 차감, 최대 40봉 보유)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
