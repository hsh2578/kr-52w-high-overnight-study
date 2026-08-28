# -*- coding: utf-8 -*-
"""변수 검증·최적화 — 지금 50개가 '익일 수익 예측'에 맞는 변수인가.

네 가지를 순서대로 답한다.

  1. **단변량 유효성** — 각 변수 하나만으로 익일 수익 순위를 맞히는가.
     지표는 일별 순위상관(IC)과 그 t값. 날짜 안에서 재는 이유는 시장 전체가
     오르내린 효과를 빼기 위해서다.
  2. **중복** — 서로 강하게 상관된 변수 군집. 상관된 변수들은 중요도를 나눠
     가져(substitution effect) 개별 중요도로는 다 하찮아 보인다.
  3. **피처 수 최적화** — 상위 k개만 쓰면 성능이 어떻게 변하는가.
     **선택은 학습 폴드 안에서만** 한다. 전체로 고르면 검정을 보고 고르는 셈이다.
  4. **PCA** — 차원 축소가 도움이 되는가. 트리는 축 정렬 분할을 쓰므로
     회전된 축이 오히려 불리할 것으로 예상되지만, 예상으로 배제하지 않고 잰다.

출력: outputs/feature_study.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from scipy.stats import spearmanr                      # noqa: E402
from sklearn.decomposition import PCA                  # noqa: E402
from sklearn.impute import SimpleImputer               # noqa: E402
from sklearn.metrics import roc_auc_score              # noqa: E402
from sklearn.preprocessing import StandardScaler       # noqa: E402

from surge_d1 import config, dataset, models, validate  # noqa: E402

LABEL = "y_exit"


def daily_ic(d: pd.DataFrame, col: str) -> tuple[float, float]:
    """일별 순위상관의 평균과 t값. 하루에 3종목 미만인 날은 건너뛴다."""
    ics = []
    for _, g in d.groupby("date"):
        if len(g) < 3 or g[col].nunique() < 3:
            continue
        r = spearmanr(g[col], g[LABEL]).statistic
        if np.isfinite(r):
            ics.append(r)
    if len(ics) < 30:
        return np.nan, np.nan
    a = np.array(ics)
    return a.mean(), a.mean() / (a.std() / np.sqrt(len(a)))


def eval_feats(df, folds, feats, pca_dim=None):
    """주어진 피처로 walk-forward 성능을 잰다. pca_dim이 있으면 폴드 안에서 PCA."""
    aucs, wins, rets = [], [], []
    for tr_m, te_m in folds:
        tr, te = df[tr_m], df[te_m]
        Xtr = tr[feats].to_numpy(float)
        Xte = te[feats].to_numpy(float)
        if pca_dim:
            # 대체·표준화·PCA 모두 **학습 폴드에서만** 적합한다.
            im = SimpleImputer(strategy="median").fit(Xtr)
            sc = StandardScaler().fit(im.transform(Xtr))
            pc = PCA(n_components=pca_dim, random_state=config.RANDOM_SEED).fit(
                sc.transform(im.transform(Xtr)))
            Xtr = pc.transform(sc.transform(im.transform(Xtr)))
            Xte = pc.transform(sc.transform(im.transform(Xte)))
        m = models._make_clf()
        m.fit(Xtr, tr["y_cls"].to_numpy(int), sample_weight=tr["weight"].to_numpy(float))
        p = m.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(te["y_cls"], p))
        k = max(1, int(len(te) * 0.2))
        top = te.iloc[np.argsort(-p)[:k]]
        wins.append(top["y_cls"].mean())
        rets.append(top[LABEL].mean())
    return np.mean(aucs), np.mean(wins) * 100, np.mean(rets) * 100


def main() -> int:
    df = dataset.load_dataset()
    feats = list(dataset.FEATURE_SETS["M4"])
    folds = validate.walk_forward_folds(df["date"], models.TEST_YEARS)
    print(f"데이터 {len(df):,}건 · 피처 {len(feats)}개 · 폴드 {len(folds)}개\n")

    # --- 1. 단변량 유효성 ---
    rows = []
    for c in feats:
        ic, t = daily_ic(df, c)
        rows.append({"feature": c, "ic": ic, "t": t,
                     "na": df[c].isna().mean() * 100,
                     "nuniq": df[c].nunique()})
    uni = pd.DataFrame(rows).sort_values("t", key=lambda s: -s.abs())

    print("=== 신호가 있는 변수 (|t| >= 2) ===")
    strong = uni[uni["t"].abs() >= 2]
    for _, r in strong.iterrows():
        print(f"  {r.feature:<24} IC {r.ic:>+7.4f}  t {r.t:>+6.2f}  결측 {r.na:>4.1f}%")
    print(f"\n  → {len(strong)}/{len(feats)}개만 단변량 신호가 있다\n")

    dead = uni[(uni["t"].abs() < 0.5) | uni["t"].isna()]
    print(f"=== 신호가 사실상 없는 변수 (|t| < 0.5): {len(dead)}개 ===")
    print("  " + ", ".join(dead["feature"].tolist()) + "\n")

    # --- 2. 중복 (상관 군집) ---
    corr = df[feats].corr(method="spearman").abs()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(feats)
             for b in feats[i + 1:] if corr.loc[a, b] >= 0.9]
    print(f"=== |상관| >= 0.9 인 쌍: {len(pairs)}개 ===")
    for a, b, v in sorted(pairs, key=lambda x: -x[2])[:10]:
        print(f"  {a:<22} ~ {b:<22} {v:.3f}")
    print()

    # --- 3. 피처 수 최적화 (선택은 학습 폴드 안에서) ---
    print("=== 피처 수 최적화 — 선택을 학습 폴드 안에서 수행 ===")
    res = []
    for k in (5, 10, 15, 20, 30, 50):
        aucs, wins, rets = [], [], []
        for tr_m, te_m in folds:
            tr, te = df[tr_m], df[te_m]
            # 이 폴드의 **학습 구간만** 보고 상위 k개를 고른다
            sc = []
            for c in feats:
                ic, t = daily_ic(tr, c)
                sc.append((abs(t) if np.isfinite(t) else 0.0, c))
            pick = [c for _, c in sorted(sc, reverse=True)[:k]]
            m = models._make_clf()
            m.fit(tr[pick].to_numpy(float), tr["y_cls"].to_numpy(int),
                  sample_weight=tr["weight"].to_numpy(float))
            p = m.predict_proba(te[pick].to_numpy(float))[:, 1]
            aucs.append(roc_auc_score(te["y_cls"], p))
            kk = max(1, int(len(te) * 0.2))
            top = te.iloc[np.argsort(-p)[:kk]]
            wins.append(top["y_cls"].mean())
            rets.append(top[LABEL].mean())
        res.append({"방식": f"상위 {k}개", "AUC": np.mean(aucs),
                    "상위20%승률": np.mean(wins) * 100,
                    "상위20%수익": np.mean(rets) * 100})
        print(f"  상위 {k:>2}개  AUC {np.mean(aucs):.4f}  "
              f"승률 {np.mean(wins)*100:5.1f}%  수익 {np.mean(rets)*100:+.3f}%", flush=True)

    a, w, r = eval_feats(df, folds, feats)
    res.append({"방식": "전체 50개", "AUC": a, "상위20%승률": w, "상위20%수익": r})
    print(f"  전체 50개  AUC {a:.4f}  승률 {w:5.1f}%  수익 {r:+.3f}%\n")

    # --- 4. PCA ---
    print("=== PCA (폴드 안에서 적합) ===")
    for dim in (5, 10, 20):
        a, w, r = eval_feats(df, folds, feats, pca_dim=dim)
        res.append({"방식": f"PCA {dim}차원", "AUC": a,
                    "상위20%승률": w, "상위20%수익": r})
        print(f"  PCA {dim:>2}차원  AUC {a:.4f}  승률 {w:5.1f}%  수익 {r:+.3f}%", flush=True)

    out = pd.DataFrame(res)
    out.to_csv(config.RESULTS_DIR / "feature_study.csv", index=False)
    uni.to_csv(config.RESULTS_DIR / "feature_univariate.csv", index=False)
    print(f"\n저장: {config.RESULTS_DIR / 'feature_study.csv'}")
    print(f"      {config.RESULTS_DIR / 'feature_univariate.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
