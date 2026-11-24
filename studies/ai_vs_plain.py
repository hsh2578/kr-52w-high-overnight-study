# -*- coding: utf-8 -*-
"""AI 신호가 '그냥 52주 신고가'보다 나은가 — 직접 비교.

이 검증이 이 프로젝트의 존재 이유다. 규칙 스크리너가 걸러낸 종목 전부를
사는 것(=그냥 돌파)과, 거기에 AI 점수를 매겨 상위만 고르는 것을 같은
기간·같은 지표로 비교한다.

중요: 원 프로젝트의 모델 비교는 **무작위 분할**이었다. 시계열에서 무작위로
나누면 미래 데이터가 학습에 섞여 성능이 부풀려진다. 여기서는 **시간순으로
잘라** 앞 기간으로 배우고 뒤 기간에서만 평가한다.

지표를 두 개 쓰는 이유:
  - max_gain_20d  : 20일 안의 **최고** 수익률. 원 프로젝트의 학습 타깃.
  - gain_at_day_20: 20일 **뒤** 실제 수익률. 최고점은 사후에만 알 수 있으므로
                    실제로 손에 쥐는 것은 이쪽에 가깝다.
둘을 나란히 보여야 "달성 가능한 수익"과 "이론적 상한"을 구분할 수 있다.

출력: outputs/ai_vs_plain.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surge_d1 import config  # noqa: E402
from lightgbm import LGBMClassifier, LGBMRegressor      # noqa: E402
from scipy.stats import ttest_ind                       # noqa: E402
from sklearn.metrics import roc_auc_score               # noqa: E402

CSV = str(config.external("screener_training.csv"))
OUT = str(Path(__file__).resolve().parents[1] / "outputs" / "ai_vs_plain.csv")

FEATS = ["breakout_pct", "volume_surge", "close_strength", "base_length",
         "volatility_contraction", "volume_dry_up", "ma200_slope",
         "pct_above_52w_low", "rs_vs_market", "market_return",
         "ma20_deviation", "liquidity", "days_since_ath", "atr_ratio"]

# 배포 모델과 같은 설정 (ml/models/52w_high/model_meta.pkl)
REG = dict(n_estimators=200, max_depth=3, learning_rate=0.01, subsample=0.8,
           reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1, n_jobs=-1)
CLF = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1)


def summarize(x: pd.DataFrame, name: str, base: pd.DataFrame | None = None) -> dict:
    """한 집단의 성과. base를 주면 그 대비 유의성까지 낸다."""
    row = {
        "구분": name,
        "픽": len(x),
        "15%+도달률": x["hit"].mean() * 100,
        "평균 최고수익": x["max_gain_20d"].mean(),
        "평균 20일후": x["gain_at_day_20"].mean(),
        "20일후 승률": (x["gain_at_day_20"] > 0).mean() * 100,
    }
    if base is not None and len(x) > 1:
        t = ttest_ind(x["gain_at_day_20"], base["gain_at_day_20"], equal_var=False)
        row["t(20일후)"] = t.statistic
        row["p"] = t.pvalue
    return row


def main() -> int:
    d = pd.read_csv(CSV)
    d["breakout_date"] = pd.to_datetime(d["breakout_date"])
    d = d.dropna(subset=FEATS + ["max_gain_20d", "gain_at_day_20"])
    d = d.sort_values("breakout_date").reset_index(drop=True)
    d["hit"] = (d["max_gain_20d"] >= 15).astype(int)   # 배포 분류기와 같은 기준

    # --- 시간순 분할: 앞 70%로 배우고 뒤 30%에서만 평가 ---
    cut = d["breakout_date"].quantile(0.70)
    tr, te = d[d["breakout_date"] <= cut], d[d["breakout_date"] > cut]
    print(f"학습 {len(tr):,}건  {tr.breakout_date.min().date()} ~ {tr.breakout_date.max().date()}")
    print(f"검정 {len(te):,}건  {te.breakout_date.min().date()} ~ {te.breakout_date.max().date()}")
    print(f"  (검정 구간은 학습에 한 번도 쓰이지 않았다)\n")

    Xtr, Xte = tr[FEATS].to_numpy(float), te[FEATS].to_numpy(float)
    clf = LGBMClassifier(**CLF).fit(Xtr, tr["hit"].to_numpy(int))
    reg = LGBMRegressor(**REG).fit(Xtr, tr["max_gain_20d"].to_numpy(float))

    prob = clf.predict_proba(Xte)[:, 1] * 100          # 성공확률 0~100
    gain = reg.predict(Xte)                            # 예상 수익률 %
    # 서비스와 동일한 AI 종합 점수 공식
    ai = prob * 0.7 + np.clip(gain * 3, 0, 100) * 0.3

    te = te.copy()
    te["ai_score"] = ai
    print(f"검정 AUC {roc_auc_score(te['hit'], prob):.4f}\n")

    rows = [summarize(te, "① 그냥 52주 돌파 (전부)")]
    for q, lab in ((0.80, "② AI 점수 상위 20%"),
                   (0.90, "③ AI 점수 상위 10%"),
                   (0.95, "④ AI 점수 상위 5%")):
        thr = te["ai_score"].quantile(q)
        rows.append(summarize(te[te["ai_score"] >= thr], lab, base=te))

    # 비교 기준선 — 단일 규칙으로 같은 수만큼 고르면?
    n20 = int(len(te) * 0.2)
    for col, lab in (("close_strength", "⑤ 마감강도 상위 20% (단일 규칙)"),
                     ("volume_surge", "⑥ 거래량 상위 20% (단일 규칙)")):
        rows.append(summarize(te.nlargest(n20, col), lab, base=te))

    res = pd.DataFrame(rows)
    print(f"{'구분':<28}{'픽':>6}{'15%+':>8}{'최고수익':>10}{'20일후':>9}"
          f"{'승률':>8}{'t':>8}{'p':>8}")
    for _, r in res.iterrows():
        t = f"{r['t(20일후)']:+.2f}" if pd.notna(r.get("t(20일후)")) else "—"
        p = f"{r['p']:.3f}" if pd.notna(r.get("p")) else "—"
        print(f"{r['구분']:<28}{int(r['픽']):>6}{r['15%+도달률']:>7.1f}%"
              f"{r['평균 최고수익']:>9.2f}%{r['평균 20일후']:>8.2f}%"
              f"{r['20일후 승률']:>7.1f}%{t:>8}{p:>8}")

    res.to_csv(OUT, index=False)
    print(f"\n저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
