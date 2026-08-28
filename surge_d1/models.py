# -*- coding: utf-8 -*-
"""M0~M5 학습·평가와 ablation 실행. 스펙 §7.1, §7.4, §7.6.

리뷰 반영 이력(Task 11):
- Important 1: evaluate()가 회귀 예측이 상수로 붕괴한 경우 ic/ic_t를
  NaN으로 낸다(0.0과 "정의되지 않음"을 CSV에서 구분).
- Important 2: 사다리 각 단계(M0→M1→...→M4, M4prime→M5)의 AUC 개선폭에
  부트스트랩 CI·p값을 붙이고, 다섯 비교 전체에 BH 다중검정 보정을 적용.
- Minor 3: isotonic 보정도 LightGBM과 동일한 sample_weight로 적합.
- Minor 4: assert_train_before_test 단위 테스트 추가(tests/test_models.py).
- 추가 지시: 학습-검정 성능 격차(과적합 진단), 폴드별 부호 일관성,
  피처 수 대비 학습 표본 크기를 ablation.csv에 기록.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold

from surge_d1 import config, dataset, features_f4, validate

# 모델 선택은 실측으로 정했다(outputs/model_comparison.csv, _model_compare.py).
# 상한가를 이벤트에서 뺀 뒤(5,675건) 10개 모델을 같은 walk-forward 4폴드로
# 비교한 결과:
#   ExtraTrees   AUC 0.5542 / 상위20% 승률 47.7% / 수익 +0.436%   <- 채택
#   RandomForest AUC 0.5503 / 48.9% / +0.418%
#   앙상블        AUC 0.5412 / 46.0% / +0.278%
#   XGBoost      AUC 0.5383 / 45.7% / +0.223%
#   LightGBM     AUC 0.5359 / 46.0% / +0.289%
#   CatBoost     AUC 0.5307 / 43.8% / +0.106%
#   LGBMRanker   AUC 0.5301 / 44.6% / +0.241%   (순위학습도 이기지 못했다)
#   Logistic     AUC 0.5167 / 41.3% / +0.086%
#   (기저)                     40.7% / +0.091%
#
# LightGBM은 다른 프로젝트에서 물려받은 것이지 이 문제에서 고른 것이 아니었다.
# 표본 5,675건에 신호가 약한 문제에서는 부스팅이 잡음을 외우기 쉽고, 분할
# 지점까지 무작위로 고르는 ExtraTrees가 분산을 더 줄여 앞선다.
# 다만 모델 간 차이(0.517~0.554)보다 **선별한다는 행위 자체**의 효과
# (기저 40.7% -> 47.7%)가 훨씬 크다는 점이 이 표의 핵심이다.
MODEL = "extratrees"          # "extratrees" | "randomforest" | "lightgbm"

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=20,
                 random_state=config.RANDOM_SEED, n_jobs=-1)
ET_PARAMS = dict(n_estimators=400, max_depth=12, min_samples_leaf=20,
                 random_state=config.RANDOM_SEED, n_jobs=-1)

REG_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.01,
                  subsample=0.8, reg_alpha=1.0, reg_lambda=5.0,
                  random_state=config.RANDOM_SEED, verbose=-1, n_jobs=-1)
CLF_PARAMS = dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=config.RANDOM_SEED, verbose=-1, n_jobs=-1)


def _make_clf():
    """분류기 하나를 새로 만든다. 폴드마다 새 인스턴스여야 상태가 안 섞인다."""
    if MODEL == "extratrees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**ET_PARAMS)
    if MODEL == "randomforest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**RF_PARAMS)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**CLF_PARAMS)


def _make_reg():
    if MODEL == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(**ET_PARAMS)
    if MODEL == "randomforest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**RF_PARAMS)
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**REG_PARAMS)

TEST_YEARS = [2023, 2024, 2025, 2026]


def assert_train_before_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """학습 최대일이 검정 최소일보다 앞서는지 검증한다.

    walk-forward를 쓰는 모든 호출 지점(run_ablation, run_leakage_comparison의
    walk_forward 분기, _catalyst_fold)에서 명시적으로 호출한다 — 스플리터
    구현을 신뢰만 하지 않고 실제로 검증한다. fit_predict 자체에는 넣지
    않는다: 셔플 K-fold 누수 비교(§7.6)는 의도적으로 이 순서를 어기는
    것이 측정 대상이고, 단위 테스트의 장난감 데이터도 시간순 분할이
    아니기 때문이다.
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


def _oof_prob(train_df: pd.DataFrame, feats: list[str],
              w: np.ndarray | None, n_splits: int = 3) -> tuple:
    """학습 구간 **안에서** 시간순 분할로 out-of-fold 확률을 만든다.

    확률 보정을 학습 예측(`clf.predict_proba(Xtr)`)에 맞추면 안 되는 이유:
    분류기는 학습 구간을 외워 AUC 0.77~0.82를 내지만 검정에서는 0.60이다.
    그 위에 isotonic을 적합하면 "raw 0.9 = 진짜 0.95"를 배우고, 검정의
    raw 0.9(실제로는 0.6쯤)에 그대로 적용해 **과신을 교정하는 대신 증폭**한다.
    실측으로 검정 Brier가 0.2848까지 올라 상수예측(0.2480)보다 나빴다.

    그래서 보정은 모델이 보지 못한 예측 위에서만 맞춘다. 분할은
    tuning.inner_time_splits와 같은 확장(expanding) 시간순 방식이라
    내부 검증에도 미래 정보가 섞이지 않는다.

    반환: (oof_prob, mask) — mask는 OOF 예측이 존재하는 행. 가장 이른
    구간은 어떤 분할에서도 검증이 되지 않아 비며, 그 행들은 보정 적합에서
    빠진다(예측 자체에는 영향 없음).
    """
    from surge_d1.tuning import inner_time_splits

    X = train_df[feats].to_numpy(dtype=float)
    y = train_df["y_cls"].to_numpy(dtype=int)
    oof = np.full(len(train_df), np.nan)
    for tr_idx, va_idx in inner_time_splits(train_df["date"], n_splits):
        if len(np.unique(y[tr_idx])) < 2:      # 한쪽 클래스뿐이면 적합 불가
            continue
        c = _make_clf()
        c.fit(X[tr_idx], y[tr_idx],
              sample_weight=None if w is None else w[tr_idx])
        oof[va_idx] = c.predict_proba(X[va_idx])[:, 1]
    return oof, ~np.isnan(oof)


def _fit_predict_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame,
                            feats: list[str], use_weight: bool = True,
                            label_reg: str = "y_raw",
                            calibrate: str = "oof"
                            ) -> tuple[dict, dict]:
    """실제 모델 적합은 한 번만 하고, 학습·검정 양쪽에 대한 예측을 함께 낸다.

    fit_predict()의 공개 계약(키 pred_reg/prob_raw/prob_cal)은 검정 쪽
    결과만 노출한다 — 학습 쪽 예측은 run_ablation의 과적합 진단
    (학습-검정 성능 격차)에만 쓴다. 재적합하지 않고 같은 reg/clf/iso로
    Xtr·Xte 양쪽을 predict하므로 추가 학습 비용은 없다.

    `calibrate`: "oof"(기본) 는 학습 구간 내부 시간순 분할의 out-of-fold
    예측으로 isotonic을 적합한다. "in_fold" 는 학습 예측에 적합하는 옛
    방식으로, 검정에서 과신을 증폭시킨다 — 회귀 테스트 전용이며 결과
    산출에 쓰지 않는다. `_oof_prob`의 독스트링에 실측 근거가 있다.
    """
    n_train, n_test = len(train_df), len(test_df)
    if not feats:
        base_ret = float(train_df[label_reg].mean())
        base_p = float(train_df["y_cls"].mean())
        train_out = {"pred_reg": np.full(n_train, base_ret),
                     "prob_raw": np.full(n_train, base_p),
                     "prob_cal": np.full(n_train, base_p)}
        test_out = {"pred_reg": np.full(n_test, base_ret),
                    "prob_raw": np.full(n_test, base_p),
                    "prob_cal": np.full(n_test, base_p)}
        return train_out, test_out

    banned = set(feats) & dataset.BANNED_AS_FEATURE
    if banned:
        raise ValueError(f"fit_predict(): 금지된 컬럼이 피처에 섞였다: {banned}")

    Xtr = train_df[feats].to_numpy(dtype=float)
    Xte = test_df[feats].to_numpy(dtype=float)
    w = train_df["weight"].to_numpy(dtype=float) if use_weight else None

    reg = _make_reg()
    reg.fit(Xtr, train_df[label_reg].to_numpy(dtype=float), sample_weight=w)
    pred_reg_tr = reg.predict(Xtr)
    pred_reg_te = reg.predict(Xte)

    clf = _make_clf()
    ytr = train_df["y_cls"].to_numpy(dtype=int)
    clf.fit(Xtr, ytr, sample_weight=w)
    prob_raw_tr = clf.predict_proba(Xtr)[:, 1]
    prob_raw_te = clf.predict_proba(Xte)[:, 1]

    # 확률 보정 (스펙 §7.4). LightGBM 두 모델은 sample_weight=w로 학습하므로
    # isotonic에도 같은 w를 넘겨 보정이 모델과 어긋나지 않게 한다(리뷰 Minor 3).
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    if calibrate == "oof":
        # 모델이 보지 못한 예측 위에서 보정한다 — 이유는 _oof_prob 참조.
        oof, m = _oof_prob(train_df, feats, w)
        if m.sum() >= 50 and len(np.unique(ytr[m])) == 2:
            iso.fit(oof[m], ytr[m], sample_weight=None if w is None else w[m])
        else:
            # OOF 표본이 너무 적으면 보정 자체가 잡음이다. 보정을 포기하고
            # 원확률을 그대로 쓴다(과신을 증폭하는 것보다 낫다).
            train_out = {"pred_reg": pred_reg_tr, "prob_raw": prob_raw_tr,
                         "prob_cal": prob_raw_tr}
            test_out = {"pred_reg": pred_reg_te, "prob_raw": prob_raw_te,
                        "prob_cal": prob_raw_te}
            return train_out, test_out
    elif calibrate == "in_fold":
        iso.fit(prob_raw_tr, ytr, sample_weight=w)
    else:
        raise ValueError(f"calibrate는 'oof' 또는 'in_fold': {calibrate!r}")
    prob_cal_tr = np.clip(iso.predict(prob_raw_tr), 0.0, 1.0)
    prob_cal_te = np.clip(iso.predict(prob_raw_te), 0.0, 1.0)

    train_out = {"pred_reg": pred_reg_tr, "prob_raw": prob_raw_tr, "prob_cal": prob_cal_tr}
    test_out = {"pred_reg": pred_reg_te, "prob_raw": prob_raw_te, "prob_cal": prob_cal_te}
    return train_out, test_out


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame,
                feats: list[str], use_weight: bool = True,
                label_reg: str = "y_raw") -> dict:
    """회귀·분류 학습 후 예측. feats가 비면 M0(상수 예측)."""
    _, test_out = _fit_predict_train_test(train_df, test_df, feats, use_weight, label_reg)
    return test_out


def evaluate(test_df: pd.DataFrame, out: dict,
             label_reg: str = "y_raw") -> dict:
    """검정(또는 학습) 구간 지표. 스펙 §7.5.

    test_df 인자는 이름과 달리 학습 구간 프레임을 넣어도 그대로
    작동한다(run_ablation이 과적합 진단을 위해 학습 구간에도 이 함수를
    재사용한다) — date/y_cls/label_reg 컬럼만 있으면 된다.

    리뷰 Important 1: 재료 표본(M4prime/M5)처럼 회귀 모델이 상수 예측으로
    붕괴하면(작은 표본 + 강한 정규화 조합) 순위상관 자체가 정의되지
    않는다. validate.spearman_ic는 이런 경우 날짜별로 nunique<2인 날을
    모두 걸러내고 최종적으로 (0.0, 0.0) 폴백을 반환하도록 설계돼 있는데,
    이를 그대로 CSV에 실으면 "측정된 상관 0"과 "애초에 측정 불가"가
    구분되지 않는다 — 플로팅 스크립트나 슬라이드가 그대로 "IC=0.000"을
    실측치처럼 보여줄 위험이 있다. 여기서 pred_reg의 전역 변동을 먼저
    확인해, 상수 예측이면 ic/ic_t를 NaN으로 남겨 그 구분이 CSV까지
    살아남게 한다.
    """
    y = test_df[label_reg].to_numpy(dtype=float)
    ycls = test_df["y_cls"].to_numpy(dtype=int)
    dates = test_df["date"]

    if len(np.unique(ycls)) < 2:
        auc = float("nan")
    else:
        # AUC는 **순위** 지표이므로 연속값 prob_raw로 잰다. 보정된 prob_cal은
        # isotonic 계단 함수라 동점이 대량 생기고, roc_auc_score는 동점을
        # 0.5로 세어 AUC를 인위적으로 깎는다(실측 폴드당 최대 -0.005).
        # 반대로 Brier는 **확률값** 지표이므로 보정된 prob_cal로 재야 한다.
        auc = float(roc_auc_score(ycls, out["prob_raw"]))
    brier = float(brier_score_loss(ycls, out["prob_cal"]))

    pred_reg = np.asarray(out["pred_reg"], dtype=float)
    if np.unique(pred_reg).size < 2:
        ic, ic_t = float("nan"), float("nan")
    else:
        ic, ic_t = validate.spearman_ic(y, pred_reg, dates)
    top = validate.top_decile_return(y, pred_reg, dates)
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

    이 exclude_key 적용은 train_df/test_df 양쪽에 동일하게 걸리므로,
    나중에 학습 구간(tr4)에 대해 과적합 진단용 지표를 낼 때도(§ 위
    evaluate 재사용) 자기포함 누수 없이 만들어진 F4를 그대로 쓰게 된다
    — 별도 조치가 필요 없다.
    """
    idx = features_f4.CatalystIndex().fit(
        train_df.loc[train_df["has_catalyst"] == 1, ["ctype", "reason", "d1"]])

    def apply(df: pd.DataFrame) -> pd.DataFrame:
        rows = [idx.transform(r.ctype, r.reason, k=10, exclude_key=r.Index)
                for r in df.itertuples(index=True)]
        feat_df = pd.DataFrame(rows, index=df.index)
        return pd.concat([df, feat_df], axis=1).reset_index(drop=True)

    return apply(train_df), apply(test_df)


def _catalyst_fold(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """재료 표본 60/40 시간 분할.

    run_ablation과 compute_ablation_deltas가 정확히 같은 분할을
    재현하도록 공유한다. 인덱스를 reset하지 않는다 — _build_f4의
    exclude_key가 원본 인덱스 라벨로 자기 자신을 찾아야 하기 때문이다.
    """
    cat = df[df["has_catalyst"] == 1]
    split = cat["date"].quantile(0.6)
    tr = cat[cat["date"] < split]
    te = cat[cat["date"] >= split]
    assert_train_before_test(tr, te)
    return tr, te


def run_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """M0~M4는 전체 표본 4폴드, M4prime/M5는 재료 표본 단일 폴드.

    각 모델·폴드마다 검정 지표뿐 아니라 같은 모델의 학습 구간 지표도
    함께 내 train_*, gap_*(=학습-검정) 컬럼으로 과적합을 직접 잰다.
    n_features/n_train도 함께 기록해 "피처 수 대비 표본 크기"를 결과표
    자체에서 읽을 수 있게 한다.
    """
    results = []

    def _row(name, fold, year, sample, feats, n_train_, test_m, train_m):
        row = {"model": name, "fold": fold, "test_year": year, "sample": sample,
              "n_features": len(feats), "n_train": n_train_, **test_m}
        for k in ("auc", "brier", "ic", "ic_t", "top_decile", "n"):
            row[f"train_{k}"] = train_m[k]
        row["gap_auc"] = train_m["auc"] - test_m["auc"]
        row["gap_ic"] = train_m["ic"] - test_m["ic"]
        row["gap_top_decile"] = train_m["top_decile"] - test_m["top_decile"]
        return row

    folds = validate.walk_forward_folds(df["date"], TEST_YEARS)
    for fi, (tr_mask, te_mask) in enumerate(folds, 1):
        tr, te = df[tr_mask].reset_index(drop=True), df[te_mask].reset_index(drop=True)
        assert_train_before_test(tr, te)
        year = int(te["date"].dt.year.mode()[0])
        for name in ("M0", "M1", "M2", "M3", "M4"):
            feats = [] if name == "M0" else dataset.FEATURE_SETS[name]
            train_out, test_out = _fit_predict_train_test(tr, te, feats)
            test_m = evaluate(te, test_out)
            train_m = evaluate(tr, train_out)
            results.append(_row(name, fi, year, "full", feats, len(tr), test_m, train_m))
            print(f"  {name} fold{fi}({year}) 완료 "
                 f"(n_feat={len(feats)}, n_train={len(tr)}, gap_auc={train_m['auc']-test_m['auc']:+.4f})",
                 flush=True)

    # 재료 표본: M4prime vs M5
    #
    # ⚠️ 표본 대비 피처가 많다. 재료가 붙은 이벤트는 약 2,150건인데 M5는
    # 피처 55개라 학습 구간이 1,000~1,300행 남짓이면 과적합이 사실상
    # 확정이다. 그렇다고 이 폴드만 하이퍼파라미터를 조이면 M4와 비교가
    # 깨진다 — **파라미터는 고정한 채 두고, 표본 크기(n)를 결과표와
    # 리포트에 크게 적어 결론의 강도를 스스로 낮춘다.**
    # M5가 M4prime을 이겨도 이 표본에서는 잠정적 증거일 뿐이다.
    cat = df[df["has_catalyst"] == 1]
    if len(cat) > 200:
        tr, te = _catalyst_fold(df)
        tr4, te4 = _build_f4(tr, te)
        for name, feats in (("M4prime", dataset.FEATURE_SETS["M4"]),
                            ("M5", dataset.FEATURE_SETS["M5"])):
            train_out, test_out = _fit_predict_train_test(tr4, te4, feats)
            test_m = evaluate(te4, test_out)
            train_m = evaluate(tr4, train_out)
            results.append(_row(name, 1, int(te["date"].dt.year.min()), "catalyst",
                                feats, len(tr4), test_m, train_m))
            print(f"  {name} (재료 표본, n_feat={len(feats)}, n_train={len(tr4)}, "
                 f"n_test={len(te4)}, gap_auc={train_m['auc']-test_m['auc']:+.4f}) 완료",
                 flush=True)

    return pd.DataFrame(results)


def _bootstrap_delta_stats(values, n: int = 2000,
                           seed: int = config.RANDOM_SEED) -> dict:
    """폴드별 델타 배열의 평균에 대한 부트스트랩 CI와 양측 p값.

    CI는 validate.bootstrap_ci(평균의 부트스트랩)를 그대로 쓴다. p값은
    같은 리샘플 절차를 동일 시드로 재현해, 리샘플된 평균이 0 이하/이상인
    비율로 낸다 — 부트스트랩 CI가 0을 포함하는지와 사실상 동치인
    양측검정이다.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"delta": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p_value": float("nan")}
    observed = float(v.mean())
    ci_lo, ci_hi = validate.bootstrap_ci(v, n=n, seed=seed)
    if v.size == 1:
        return {"delta": observed, "ci_lo": ci_lo, "ci_hi": ci_hi,
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    frac_le0 = float((means <= 0).mean())
    frac_ge0 = float((means >= 0).mean())
    p = float(min(1.0, 2 * min(frac_le0, frac_ge0)))
    return {"delta": observed, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": p}


def bootstrap_paired_auc_delta(y_cls, prob_a, prob_b, n: int = 2000,
                               seed: int = config.RANDOM_SEED) -> dict:
    """짝지은(paired) 행 리샘플로 AUC 차이(b-a)의 분포를 낸다.

    AUC는 행별 값의 평균으로 분해되지 않는 순위 통계라
    validate.bootstrap_ci(평균의 부트스트랩)를 그대로 적용할 수 없다.
    대신 같은 인덱스로 두 모델의 예측·정답을 함께 리샘플해 AUC를 각각
    다시 계산하고, 그 차이의 분포에서 백분위수 CI와 부트스트랩 양측
    p값을 얻는다. M4prime→M5처럼 폴드가 하나뿐인 비교에 쓴다.
    """
    y = np.asarray(y_cls, dtype=int)
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    m = len(y)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            continue
        diffs.append(float(roc_auc_score(yi, b[idx]) - roc_auc_score(yi, a[idx])))
    diffs = np.asarray(diffs, dtype=float)
    ci_lo, ci_hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    frac_a_better = float((diffs < 0).mean())
    frac_b_better = float((diffs > 0).mean())
    p = float(min(1.0, 2 * min(frac_a_better, frac_b_better)))
    observed = float(roc_auc_score(y, b) - roc_auc_score(y, a))
    return {"delta": observed, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": p,
            "frac_a_better": frac_a_better, "n_boot": int(diffs.size)}


def compute_ablation_deltas(df: pd.DataFrame, ablation_df: pd.DataFrame) -> pd.DataFrame:
    """사다리(M0→M1→...→M4)와 M4prime→M5, 각 단계에 부트스트랩 CI·p값·
    폴드별 부호 일관성을 붙인다. 스펙 §7.6이 요구하는 "개선에는 불확실성이
    따라와야 한다"를 AUC(주 지표) 기준으로 구현한다.

    BH 다중검정 보정(validate.benjamini_hochberg)은 이 다섯 AUC 비교
    전체에만 적용한다 — 다섯 번 검정하면 우연히 하나는 유의하게 나올 수
    있어서다. IC 델타는 참고용으로 같은 표에 남기되(리뷰가 지적한
    M2→M3의 IC 악화를 결과표에서 바로 보이게 하려는 목적) BH 보정
    대상(5개 AUC 검정)에는 포함하지 않는다 — 별도 가족이다.
    """
    full = ablation_df[ablation_df["sample"] == "full"]
    ladder = [("M0", "M1"), ("M1", "M2"), ("M2", "M3"), ("M3", "M4")]
    rows = []

    for a, b in ladder:
        pa = full[full["model"] == a].set_index("fold")["auc"].sort_index()
        pb = full[full["model"] == b].set_index("fold")["auc"].sort_index()
        common = pa.index.intersection(pb.index)
        fold_deltas = pb.loc[common] - pa.loc[common]
        stats = _bootstrap_delta_stats(fold_deltas.to_numpy())
        n_pos, n_tot = int((fold_deltas > 0).sum()), int(fold_deltas.size)
        rows.append({
            "step": f"{a}->{b}", "metric": "auc", "n_fold": n_tot, "n_row": None,
            "delta": stats["delta"], "ci_lo": stats["ci_lo"], "ci_hi": stats["ci_hi"],
            "p_value_raw": stats["p_value"],
            "sign_pos": n_pos, "sign_total": n_tot, "sign_ratio": f"{n_pos}/{n_tot}",
            "fold_deltas": ";".join(f"fold{k}:{v:+.4f}" for k, v in fold_deltas.items()),
        })

        # 참고용 IC 델타 — BH 보정 대상에는 넣지 않는다(별도 가족).
        pa_ic = full[full["model"] == a].set_index("fold")["ic"].sort_index()
        pb_ic = full[full["model"] == b].set_index("fold")["ic"].sort_index()
        common_ic = pa_ic.index.intersection(pb_ic.index)
        ic_deltas = (pb_ic.loc[common_ic] - pa_ic.loc[common_ic]).dropna()
        if len(ic_deltas):
            ic_stats = _bootstrap_delta_stats(ic_deltas.to_numpy())
            n_pos_ic, n_tot_ic = int((ic_deltas > 0).sum()), int(ic_deltas.size)
            rows.append({
                "step": f"{a}->{b}", "metric": "ic", "n_fold": n_tot_ic, "n_row": None,
                "delta": ic_stats["delta"], "ci_lo": ic_stats["ci_lo"],
                "ci_hi": ic_stats["ci_hi"], "p_value_raw": ic_stats["p_value"],
                "sign_pos": n_pos_ic, "sign_total": n_tot_ic,
                "sign_ratio": f"{n_pos_ic}/{n_tot_ic}",
                "fold_deltas": ";".join(f"fold{k}:{v:+.4f}" for k, v in ic_deltas.items()),
            })

    # 재료 표본: 폴드가 하나뿐이라 위(폴드 간) 부트스트랩을 못 쓴다 —
    # 대신 행 단위 짝지은 부트스트랩을 쓴다. run_ablation과 동일한 분할을
    # 재현해 원 예측값을 다시 얻는다(재학습 비용은 전체 실행시간 대비 작다).
    tr, te = _catalyst_fold(df)
    tr4, te4 = _build_f4(tr, te)
    _, out_prime = _fit_predict_train_test(tr4, te4, dataset.FEATURE_SETS["M4"])
    _, out_m5 = _fit_predict_train_test(tr4, te4, dataset.FEATURE_SETS["M5"])
    ycls = te4["y_cls"].to_numpy(dtype=int)
    cat_stats = bootstrap_paired_auc_delta(ycls, out_prime["prob_cal"], out_m5["prob_cal"])
    p_perm = validate.permutation_pvalue(
        lambda y, p, d: roc_auc_score(y, p), ycls,
        out_prime["prob_cal"], out_m5["prob_cal"], te4["date"],
        n=2000, seed=config.RANDOM_SEED)
    rows.append({
        "step": "M4prime->M5", "metric": "auc", "n_fold": None, "n_row": int(len(ycls)),
        "delta": cat_stats["delta"], "ci_lo": cat_stats["ci_lo"], "ci_hi": cat_stats["ci_hi"],
        "p_value_raw": cat_stats["p_value"],
        "sign_pos": None, "sign_total": None,
        "sign_ratio": f"부트스트랩 중 M5 우위 비율={1 - cat_stats['frac_a_better']:.3f}",
        "fold_deltas": f"n_boot={cat_stats['n_boot']};perm_p_onesided={p_perm:.4f}",
    })

    out = pd.DataFrame(rows)
    auc_mask = out["metric"] == "auc"
    out["p_value_bh"] = np.nan
    out.loc[auc_mask, "p_value_bh"] = validate.benjamini_hochberg(
        out.loc[auc_mask, "p_value_raw"].tolist())
    return out


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

    print("\n=== 학습-검정 성능 격차 (과적합 진단) ===")
    gap_full = ab[ab["sample"] == "full"].groupby("model")[
        ["n_features", "auc", "train_auc", "gap_auc", "ic", "train_ic", "gap_ic"]].mean()
    print(gap_full.to_string())
    gap_cat = ab[ab["sample"] == "catalyst"].set_index("model")[
        ["n_features", "n_train", "auc", "train_auc", "gap_auc"]]
    print(gap_cat.to_string())

    print("\n=== ablation 사다리 단계별 델타 (부트스트랩 CI·p값·BH 보정) ===")
    deltas = compute_ablation_deltas(df, ab)
    deltas.to_csv(config.RESULTS_DIR / "ablation_deltas.csv", index=False, encoding="utf-8-sig")
    print(deltas.to_string(index=False))

    print("\n=== 누수 편향 비교 ===")
    lk = run_leakage_comparison(df)
    lk.to_csv(config.RESULTS_DIR / "leakage.csv", index=False, encoding="utf-8-sig")
    print(lk.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
