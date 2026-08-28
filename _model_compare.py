# -*- coding: utf-8 -*-
"""모델 비교 — LightGBM이 이 문제에 정말 맞는 선택인가.

배경: LightGBM은 다른 프로젝트에서 성능이 좋아 물려받은 것이지, 이 문제에서
비교해 고른 것이 아니었다. 그 지적이 정당하므로 실측한다.

같은 데이터(M4 50개 피처)·같은 walk-forward 4폴드·같은 표본가중치로
다섯 모델을 돌린다. 판단 기준은 AUC 하나가 아니라 **상위 20% 픽의 실제
승률·수익** — 이 연구의 목적이 '고르는 것'이기 때문이다.

출력: outputs/model_comparison.csv
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.ensemble import RandomForestClassifier          # noqa: E402
from sklearn.impute import SimpleImputer                     # noqa: E402
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import roc_auc_score                    # noqa: E402
from sklearn.pipeline import make_pipeline                   # noqa: E402
from sklearn.preprocessing import StandardScaler             # noqa: E402

from surge_d1 import dataset, models, validate               # noqa: E402


def has(m: str) -> bool:
    try:
        __import__(m)
        return True
    except ImportError:
        return False


def build(name: str):
    """모델 하나를 새로 만든다. 폴드마다 새 인스턴스를 써야 상태가 안 섞인다."""
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**dict(models.CLF_PARAMS, n_jobs=1))
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=1, eval_metric="logloss")
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_depth=10,
                                      min_samples_leaf=20, random_state=42, n_jobs=-1)
    if name == "ExtraTrees":
        # RandomForest보다 분할 지점까지 무작위로 고른다. 잡음이 큰 데이터에서
        # 분산을 더 줄여주는 경우가 많아 후보로 넣는다.
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=400, max_depth=12,
                                    min_samples_leaf=20, random_state=42, n_jobs=-1)
    if name == "HistGradientBoosting":
        # sklearn의 히스토그램 부스팅. 결측을 분기로 직접 학습한다.
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=1.0, random_state=42)
    if name == "CatBoost":
        # ordered boosting이 작은·잡음 많은 표본의 타깃 누수를 줄이도록 설계돼
        # 있어, 이 규모(6천건)에서 유력한 후보다.
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                                  l2_leaf_reg=6.0, random_seed=42, verbose=0,
                                  allow_writing_files=False, thread_count=4)
    if name == "Ensemble(RF+XGB+Cat)":
        # 서로 다른 편향을 평균내면 보통 작지만 실질적인 이득이 있다.
        from sklearn.ensemble import VotingClassifier
        return VotingClassifier(
            estimators=[("rf", build("RandomForest")),
                        ("xgb", build("XGBoost")),
                        ("cat", build("CatBoost"))],
            voting="soft", n_jobs=1)
    if name == "LogisticRegression":
        # 선형모형은 결측·스케일을 스스로 다루지 못하므로 파이프라인으로 감싼다.
        # 대체값은 **학습 폴드에서만** 적합된다(파이프라인이 보장).
        return make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(),
                             LogisticRegression(max_iter=3000, C=1.0))
    raise ValueError(name)


# --- 순위학습 (Learning-to-Rank) ---
# 이 연구의 목적은 "그날 돌파한 종목들 **중에서** 고르기"다. 회귀·분류는 각
# 행을 독립으로 보지만, LambdaRank는 **같은 날 안의 순서**를 직접 최적화한다.
# 그룹 = 거래일. 금융 횡단면 예측에서 표준적으로 쓰이는 틀이다(qlib 등).
RANKERS = {"LGBMRanker", "XGBRanker"}


def fit_predict_ranker(name, tr, te, feats):
    """날짜를 그룹으로 묶어 순위학습하고 검정 점수를 돌려준다."""
    tr = tr.sort_values("date")
    Xtr = tr[feats].to_numpy(float)
    ytr = tr["y_cls"].to_numpy(int)          # 관련도 0/1
    grp = tr.groupby("date").size().to_numpy()
    Xte = te[feats].to_numpy(float)
    if name == "LGBMRanker":
        from lightgbm import LGBMRanker
        m = LGBMRanker(n_estimators=300, max_depth=6, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=1,
                       random_state=42, verbose=-1,
                       objective="lambdarank", label_gain=[0, 1])
        m.fit(Xtr, ytr, group=grp)
    else:
        from xgboost import XGBRanker
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        Xte = np.where(np.isfinite(Xte), Xte, med)
        m = XGBRanker(n_estimators=300, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=1,
                      random_state=42, objective="rank:pairwise")
        m.fit(Xtr, ytr, group=grp)
    return m.predict(Xte)

def main() -> int:
    df = dataset.load_dataset()
    feats = dataset.FEATURE_SETS["M4"]
    folds = validate.walk_forward_folds(df["date"], models.TEST_YEARS)
    print(f"데이터 {len(df):,}건 · 피처 {len(feats)}개 · 폴드 {len(folds)}개\n")

    names = ["LightGBM", "RandomForest", "ExtraTrees",
             "HistGradientBoosting", "LogisticRegression"]
    if has("xgboost"):
        names.insert(1, "XGBoost")
    if has("catboost"):
        names.insert(2, "CatBoost")
    if has("xgboost") and has("catboost"):
        names.append("Ensemble(RF+XGB+Cat)")
    names += ["LGBMRanker"] + (["XGBRanker"] if has("xgboost") else [])

    rows = []
    for name in names:
        t0 = time.time()
        aucs, wins, rets, n_pick = [], [], [], 0
        for tr_m, te_m in folds:
            tr, te = df[tr_m], df[te_m]
            Xtr = tr[feats].to_numpy(float)
            Xte = te[feats].to_numpy(float)
            ytr = tr["y_cls"].to_numpy(int)
            w = tr["weight"].to_numpy(float)

            if name in RANKERS:
                p = fit_predict_ranker(name, tr, te, feats)
                aucs.append(roc_auc_score(te["y_cls"], p))
                k = max(1, int(len(te) * 0.2))
                top = te.iloc[np.argsort(-p)[:k]]
                wins.append(top["y_cls"].mean()); rets.append(top["y_raw"].mean())
                n_pick += len(top)
                continue

            if name in ("CatBoost", "Ensemble(RF+XGB+Cat)"):
                # CatBoost는 NaN을 받지만 VotingClassifier가 검증에서 막는다.
                # 두 경우 모두 **학습 폴드** 중앙값으로 채워 조건을 맞춘다.
                med = np.nanmedian(Xtr, axis=0)
                med = np.where(np.isfinite(med), med, 0.0)
                Xtr = np.where(np.isfinite(Xtr), Xtr, med)
                Xte = np.where(np.isfinite(Xte), Xte, med)
            m = build(name)
            # 파이프라인은 sample_weight를 최종 단계 이름으로 넘겨야 한다
            # (Pipeline.fit(X, y, logisticregression__sample_weight=...)).
            # 가중치를 조용히 버리면 모델마다 다른 조건으로 비교하게 된다.
            if hasattr(m, "steps"):
                last = m.steps[-1][0]
                m.fit(Xtr, ytr, **{f"{last}__sample_weight": w})
            else:
                m.fit(Xtr, ytr, sample_weight=w)
            p = m.predict_proba(Xte)[:, 1]

            aucs.append(roc_auc_score(te["y_cls"], p))
            # 상위 20% 픽 — '고르는' 목적에 직접 대응하는 지표
            k = max(1, int(len(te) * 0.2))
            top = te.iloc[np.argsort(-p)[:k]]
            wins.append(top["y_cls"].mean())
            rets.append(top["y_raw"].mean())
            n_pick += len(top)

        rows.append({"model": name, "auc": np.mean(aucs),
                     "auc_min": np.min(aucs), "auc_max": np.max(aucs),
                     "top20_win": np.mean(wins) * 100,
                     "top20_ret": np.mean(rets) * 100,
                     "n_pick": n_pick, "sec": time.time() - t0})
        print(f"  {name:<20} AUC {np.mean(aucs):.4f}  "
              f"상위20% 승률 {np.mean(wins)*100:5.1f}%  "
              f"수익 {np.mean(rets)*100:+.3f}%  ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows).sort_values("auc", ascending=False)
    te_all = pd.concat([df[m] for _, m in folds])
    bw, br = te_all["y_cls"].mean() * 100, te_all["y_raw"].mean() * 100

    print(f"\n{'모델':<22}{'AUC':>8}{'폴드범위':>18}{'상위20%승률':>13}{'상위20%수익':>13}")
    for _, x in res.iterrows():
        print(f"{x['model']:<22}{x['auc']:>8.4f}"
              f"{x['auc_min']:>9.3f}~{x['auc_max']:<8.3f}"
              f"{x['top20_win']:>12.1f}%{x['top20_ret']:>12.3f}%")
    print(f"{'(기저 · 전체 검정)':<22}{'—':>8}{'—':>18}{bw:>12.1f}%{br:>12.3f}%")

    out = dataset.config.RESULTS_DIR / "model_comparison.csv"
    res.assign(base_win=bw, base_ret=br).to_csv(out, index=False)
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
