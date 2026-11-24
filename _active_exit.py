# -*- coding: utf-8 -*-
"""능동 매매 관점 — 목표수익 청산 규칙으로 AI 신호가 실제로 유리한가.

배경: 학습 타깃이 '20일 내 최고 수익률'이라 "잡을 수 없는 값"이라고 봤는데,
모의투자대회처럼 **능동적으로 지켜보다 파는** 용도라면 이야기가 다르다.
최고점을 정확히 맞출 수는 없어도, **목표가에 닿으면 판다**는 규칙은 실행 가능하다.

그래서 세 가지를 나란히 잰다:
  - 20일 후 실제      : 아무 규칙 없이 20일 들고 있기 (하한)
  - 목표수익 청산      : 종가가 목표에 닿으면 그날 청산, 아니면 20일째 청산 (실현 가능)
  - 20일 내 최고      : 완벽한 타이밍 (상한, 달성 불가)

목표 도달 판정에 **종가**를 쓴다. 장중 고가로 판정하면 그 가격에 실제로 팔았다는
보장이 없어 결과가 낙관적으로 부풀려진다.

출력: outputs/active_exit.csv
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
OUT = r"C:\Users\hsh\Desktop\03_투자·코딩 프로젝트\급등주 분석프로그램\outputs\active_exit.csv"

FEATS = ["breakout_pct", "volume_surge", "close_strength", "base_length",
         "volatility_contraction", "volume_dry_up", "ma200_slope",
         "pct_above_52w_low", "rs_vs_market", "market_return",
         "ma20_deviation", "liquidity", "days_since_ath", "atr_ratio"]
REG = dict(n_estimators=200, max_depth=3, learning_rate=0.01, subsample=0.8,
           reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1, n_jobs=-1)
CLF = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1)
COST = 0.21          # 왕복 거래비용 %
HOLD = 20


def simulate(df: pd.DataFrame, i: int, target: float, stop: float) -> float:
    """봉 i 종가 매수. 이후 20봉 동안 종가 기준으로 목표/손절을 확인한다.

    목표와 손절이 같은 날 함께 걸리는 일은 종가 하나만 보므로 생기지 않는다.
    끝까지 아무 것도 안 걸리면 20봉째 종가에 청산한다.
    """
    c0 = float(df["Close"].iloc[i])
    if not np.isfinite(c0) or c0 <= 0:
        return np.nan
    end = min(i + HOLD, len(df) - 1)
    for j in range(i + 1, end + 1):
        c = float(df["Close"].iloc[j])
        if not np.isfinite(c) or c <= 0:
            continue
        r = (c / c0 - 1) * 100
        if r >= target or r <= -stop:
            return r - COST
    c = float(df["Close"].iloc[end])
    return (c / c0 - 1) * 100 - COST if np.isfinite(c) and c > 0 else np.nan


def main() -> int:
    d = pd.read_csv(CSV)
    d["breakout_date"] = pd.to_datetime(d["breakout_date"])
    d["ticker"] = d["ticker"].astype(str).str.zfill(6)
    d = d.dropna(subset=FEATS + ["max_gain_20d", "gain_at_day_20"])
    d = d.sort_values("breakout_date").reset_index(drop=True)
    d["hit"] = (d["max_gain_20d"] >= 15).astype(int)

    cut = d["breakout_date"].quantile(0.70)
    tr, te = d[d["breakout_date"] <= cut], d[d["breakout_date"] > cut].copy()
    print(f"학습 {len(tr):,}건 → 검정 {len(te):,}건 "
          f"({te.breakout_date.min().date()} ~ {te.breakout_date.max().date()})\n")

    Xtr = tr[FEATS].to_numpy(float)
    clf = LGBMClassifier(**CLF).fit(Xtr, tr["hit"].to_numpy(int))
    reg = LGBMRegressor(**REG).fit(Xtr, tr["max_gain_20d"].to_numpy(float))
    Xte = te[FEATS].to_numpy(float)
    prob = clf.predict_proba(Xte)[:, 1] * 100
    gain = reg.predict(Xte)
    te["ai_score"] = prob * 0.7 + np.clip(gain * 3, 0, 100) * 0.3

    # --- 청산 규칙 시뮬레이션 ---
    ohlcv = sources.load_ohlcv()
    rules = [(10.0, 7.0), (15.0, 10.0), (20.0, 10.0)]
    for t_, s_ in rules:
        te[f"exit_{int(t_)}"] = np.nan
    miss = 0
    for idx, r in te.iterrows():
        df = ohlcv.get(r["ticker"])
        if df is None:
            miss += 1
            continue
        pos = df.index.get_indexer([r["breakout_date"]])[0]
        if pos < 0 or pos + 1 >= len(df):
            miss += 1
            continue
        for t_, s_ in rules:
            te.at[idx, f"exit_{int(t_)}"] = simulate(df, pos, t_, s_)
    print(f"경로 시뮬레이션: {len(te)-miss:,}건 매칭 (실패 {miss})\n")

    def block(x, name, base=None):
        row = {"구분": name, "픽": len(x),
               "20일후": x["gain_at_day_20"].mean(),
               "20일내최고": x["max_gain_20d"].mean()}
        for t_, _ in rules:
            row[f"목표{int(t_)}%"] = x[f"exit_{int(t_)}"].mean()
        if base is not None and len(x) > 1:
            tt = ttest_ind(x["exit_15"].dropna(), base["exit_15"].dropna(),
                           equal_var=False)
            row["t(목표15)"] = tt.statistic
            row["p"] = tt.pvalue
        return row

    rows = [block(te, "① 그냥 52주 돌파 (전부)")]
    for q, lab in ((0.80, "② AI 상위 20%"), (0.90, "③ AI 상위 10%"),
                   (0.95, "④ AI 상위 5%")):
        thr = te["ai_score"].quantile(q)
        rows.append(block(te[te["ai_score"] >= thr], lab, base=te))
    n20 = int(len(te) * 0.2)
    rows.append(block(te.nlargest(n20, "close_strength"),
                      "⑤ 마감강도 상위 20%", base=te))

    res = pd.DataFrame(rows)
    print(f"{'구분':<26}{'픽':>6}{'20일후':>9}{'최고':>9}"
          f"{'목표10%':>10}{'목표15%':>10}{'목표20%':>10}{'t':>8}{'p':>8}")
    for _, r in res.iterrows():
        t = f"{r['t(목표15)']:+.2f}" if pd.notna(r.get("t(목표15)")) else "—"
        p = f"{r['p']:.3f}" if pd.notna(r.get("p")) else "—"
        print(f"{r['구분']:<26}{int(r['픽']):>6}{r['20일후']:>8.2f}%"
              f"{r['20일내최고']:>8.2f}%{r['목표10%']:>9.2f}%"
              f"{r['목표15%']:>9.2f}%{r['목표20%']:>9.2f}%{t:>8}{p:>8}")

    print("\n(목표N% = 종가가 +N%에 닿으면 그날 청산, 손절 -7~10%, "
          "아니면 20일째 청산. 왕복비용 0.21% 차감)")
    res.to_csv(OUT, index=False)
    print(f"저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
