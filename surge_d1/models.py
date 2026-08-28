# -*- coding: utf-8 -*-
"""M0~M5 학습·평가와 ablation 실행. 스펙 §7.1, §7.4, §7.6."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold

from surge_d1 import config, dataset, features_f4, validate

REG_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.01,
                  subsample=0.8, reg_alpha=1.0, reg_lambda=5.0,
                  random_state=config.RANDOM_SEED, verbose=-1, n_jobs=-1)
CLF_PARAMS = dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=config.RANDOM_SEED, verbose=-1, n_jobs=-1)

TEST_YEARS = [2023, 2024, 2025, 2026]


def assert_train_before_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """학습 최대일이 검정 최소일보다 앞서는지 검증한다.

    walk-forward를 쓰는 모든 호출 지점(run_ablation, run_leakage_comparison의
    walk_forward 분기)에서 명시적으로 호출한다 — 스플리터 구현을 신뢰만
    하지 않고 실제로 검증한다. fit_predict 자체에는 넣지 않는다: 셔플
    K-fold 누수 비교(§7.6)는 의도적으로 이 순서를 어기는 것이 측정
    대상이고, 단위 테스트의 장난감 데이터도 시간순 분할이 아니기 때문이다.
    """
    if len(train_df) == 0 or len(test_df) == 0:
        return
    max_tr_date = pd.to_datetime(train_df["date"]).max()
    min_te_date = pd.to_datetime(test_df["date"]).min()
    if max_tr_date >= min_te_date:
        raise ValueError(
            f"학습 최대일({max_tr_date.date()})이 검정 최소일"
            f"({min_te_date.date()})보다 앞서지 않는다 — 미래정보 누수 가능성."
        )


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame,
                feats: list[str], use_weight: bool = True,
                label_reg: str = "y_raw") -> dict:
    """회귀·분류 학습 후 예측. feats가 비면 M0(상수 예측)."""
    n_test = len(test_df)
    if not feats:
        base_ret = float(train_df[label_reg].mean())
        base_p = float(train_df["y_cls"].mean())
        return {"pred_reg": np.full(n_test, base_ret),
                "prob_raw": np.full(n_test, base_p),
                "prob_cal": np.full(n_test, base_p)}

    banned = set(feats) & dataset.BANNED_AS_FEATURE
    if banned:
        raise ValueError(f"fit_predict(): 금지된 컬럼이 피처에 섞였다: {banned}")

    from lightgbm import LGBMClassifier, LGBMRegressor

    Xtr = train_df[feats].to_numpy(dtype=float)
    Xte = test_df[feats].to_numpy(dtype=float)
    w = train_df["weight"].to_numpy(dtype=float) if use_weight else None

    reg = LGBMRegressor(**REG_PARAMS)
    reg.fit(Xtr, train_df[label_reg].to_numpy(dtype=float), sample_weight=w)
    pred_reg = reg.predict(Xte)

    clf = LGBMClassifier(**CLF_PARAMS)
    ytr = train_df["y_cls"].to_numpy(dtype=int)
    clf.fit(Xtr, ytr, sample_weight=w)
    prob_raw = clf.predict_proba(Xte)[:, 1]

    # 확률 보정: 학습 구간 내부 예측으로 isotonic을 적합 (스펙 §7.4)
    prob_tr = clf.predict_proba(Xtr)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(prob_tr, ytr)
    prob_cal = np.clip(iso.predict(prob_raw), 0.0, 1.0)

    return {"pred_reg": pred_reg, "prob_raw": prob_raw, "prob_cal": prob_cal}


def evaluate(test_df: pd.DataFrame, out: dict,
             label_reg: str = "y_raw") -> dict:
    """검정 구간 지표. 스펙 §7.5."""
    y = test_df[label_reg].to_numpy(dtype=float)
    ycls = test_df["y_cls"].to_numpy(dtype=int)
    dates = test_df["date"]

    if len(np.unique(ycls)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(ycls, out["prob_cal"]))
    brier = float(brier_score_loss(ycls, out["prob_cal"]))
    ic, ic_t = validate.spearman_ic(y, out["pred_reg"], dates)
    top = validate.top_decile_return(y, out["pred_reg"], dates)
    return {"auc": auc, "brier": brier, "ic": ic, "ic_t": ic_t,
            "top_decile": top, "n": int(len(test_df))}


def _build_f4(train_df: pd.DataFrame, test_df: pd.DataFrame
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """폴드 학습 구간만으로 CatalystIndex를 적합해 F4를 만든다. 스펙 §5.4.

    ⚠️ 자기포함 누수 방어: train_df/test_df의 인덱스를 절대 reset하지
    않는다 — CatalystIndex.fit()이 그 인덱스 라벨을 "키"로 보존하고,
    transform(exclude_key=row_index)가 그 라벨로 자기 자신을 후보 풀·
    유형 평균에서 제외한다. 학습 행이든 검정 행이든 예외 없이 매 행마다
    exclude_key=그 행 자신의 인덱스 라벨을 넘긴다(features_f4.py의
    "Task 11 호출 규약" 참조) — 학습 행에서 이를 빠뜨리면 예외도 경고도
    없이 그 행 자신의 d1이 analog_mean_d1/type_prior로 새어 들어온다.
    """
    idx = features_f4.CatalystIndex().fit(
        train_df.loc[train_df["has_catalyst"] == 1, ["ctype", "reason", "d1"]])

    def apply(df: pd.DataFrame) -> pd.DataFrame:
        rows = [idx.transform(r.ctype, r.reason, k=10, exclude_key=r.Index)
                for r in df.itertuples(index=True)]
        feat_df = pd.DataFrame(rows, index=df.index)
        return pd.concat([df, feat_df], axis=1).reset_index(drop=True)

    return apply(train_df), apply(test_df)


def run_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """M0~M4는 전체 표본 4폴드, M4prime/M5는 재료 표본 단일 폴드."""
    results = []

    folds = validate.walk_forward_folds(df["date"], TEST_YEARS)
    for fi, (tr_mask, te_mask) in enumerate(folds, 1):
        tr, te = df[tr_mask].reset_index(drop=True), df[te_mask].reset_index(drop=True)
        assert_train_before_test(tr, te)
        year = int(te["date"].dt.year.mode()[0])
        for name in ("M0", "M1", "M2", "M3", "M4"):
            feats = [] if name == "M0" else dataset.FEATURE_SETS[name]
            out = fit_predict(tr, te, feats)
            results.append({"model": name, "fold": fi, "test_year": year,
                            "sample": "full", **evaluate(te, out)})
            print(f"  {name} fold{fi}({year}) 완료", flush=True)

    # 재료 표본: M4prime vs M5
    #
    # ⚠️ 표본 대비 피처가 많다. 재료가 붙은 이벤트는 약 2,150건인데 M5는
    # 피처 55개라 학습 구간이 1,000~1,300행 남짓이면 과적합이 사실상
    # 확정이다. 그렇다고 이 폴드만 하이퍼파라미터를 조이면 M4와 비교가
    # 깨진다 — **파라미터는 고정한 채 두고, 표본 크기(n)를 결과표와
    # 리포트에 크게 적어 결론의 강도를 스스로 낮춘다.**
    # M5가 M4prime을 이겨도 이 표본에서는 잠정적 증거일 뿐이다.
    #
    # 인덱스를 reset하지 않는다 — _build_f4가 exclude_key로 자기 자신을
    # 배제하려면 원본 df의 유일한 행 인덱스가 그대로 보존돼야 한다.
    cat = df[df["has_catalyst"] == 1]
    if len(cat) > 200:
        split = cat["date"].quantile(0.6)
        tr = cat[cat["date"] < split]
        te = cat[cat["date"] >= split]
        assert_train_before_test(tr, te)
        tr4, te4 = _build_f4(tr, te)
        for name, feats in (("M4prime", dataset.FEATURE_SETS["M4"]),
                            ("M5", dataset.FEATURE_SETS["M5"])):
            out = fit_predict(tr4, te4, feats)
            results.append({"model": name, "fold": 1,
                            "test_year": int(te["date"].dt.year.min()),
                            "sample": "catalyst", **evaluate(te4, out)})
            print(f"  {name} (재료 표본, n_train={len(tr4)}, n_test={len(te4)}) 완료",
                 flush=True)

    return pd.DataFrame(results)


def run_leakage_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """같은 데이터를 셔플 CV와 walk-forward로 평가해 편향을 잰다. 스펙 §7.6."""
    feats = dataset.FEATURE_SETS["M1"]
    rows = []

    # 셔플 K-fold는 이전 프로젝트(ml/src/final_model.py)의 방법론을 그대로
    # 재현한다 — 시계열인데도 무작위로 섞어 학습/검정을 나누므로, 미래
    # 데이터가 학습에 섞여 들어가는 것이 바로 측정 대상인 편향이다.
    # assert_train_before_test를 여기서는 의도적으로 호출하지 않는다.
    kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    aucs, ics = [], []
    for tr_i, te_i in kf.split(df):
        tr = df.iloc[tr_i].reset_index(drop=True)
        te = df.iloc[te_i].reset_index(drop=True)
        m = evaluate(te, fit_predict(tr, te, feats))
        aucs.append(m["auc"])
        ics.append(m["ic"])
    rows.append({"method": "shuffle_kfold", "auc": float(np.nanmean(aucs)),
                 "ic": float(np.nanmean(ics))})

    aucs, ics = [], []
    for tr_mask, te_mask in validate.walk_forward_folds(df["date"], TEST_YEARS):
        tr = df[tr_mask].reset_index(drop=True)
        te = df[te_mask].reset_index(drop=True)
        assert_train_before_test(tr, te)
        m = evaluate(te, fit_predict(tr, te, feats))
        aucs.append(m["auc"])
        ics.append(m["ic"])
    rows.append({"method": "walk_forward", "auc": float(np.nanmean(aucs)),
                 "ic": float(np.nanmean(ics))})

    out = pd.DataFrame(rows)
    out["auc_bias"] = out["auc"] - out["auc"].iloc[1]
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_parquet(config.DATASET_PATH)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== ablation (M0~M5) ===")
    ab = run_ablation(df)
    ab.to_csv(config.RESULTS_DIR / "ablation.csv", index=False, encoding="utf-8-sig")
    print(ab.groupby("model")[["auc", "ic", "top_decile"]].mean().to_string())

    print("\n=== 누수 편향 비교 ===")
    lk = run_leakage_comparison(df)
    lk.to_csv(config.RESULTS_DIR / "leakage.csv", index=False, encoding="utf-8-sig")
    print(lk.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
