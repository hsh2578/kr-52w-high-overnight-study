# -*- coding: utf-8 -*-
"""Task 15 — 하이퍼파라미터·전처리 민감도 실험.

**왜 필요한가**: 이 연구의 핵심 결론("ML이 `close_strength` 단일 지표 대비 ~7%밖에
못 더한다")은 `models.REG_PARAMS`/`CLF_PARAMS`(20일 지평·셔플 CV로 튜닝된 이전
프로젝트 값) 위에서, 전처리 없이 나온 것이다. `CLF_PARAMS`는 정규화가 전무해
학습 AUC가 0.77~0.82까지 오르고(검정 대비 격차 ~0.20), `REG_PARAMS`는 반대로
과정규화되어 소표본에서 상수로 붕괴한다. 이 모듈은 그 결론이 "모델을 제대로
안 맞춰서" 나온 것인지를 검증한다.

**핵심 원칙 — 튜닝은 학습 폴드 안에서만 한다.** 검정(test) 구간 결과를 보고
하이퍼파라미터를 고르면 체리피킹이다. 이 모듈의 모든 튜닝(`inner_tune`)은
그 폴드의 **학습 프레임만**을 입력받아, 그 안에서 다시 시간순으로 쪼갠
내부 검증(`inner_time_splits`)으로 최적점을 고른다 — 검정 데이터는 단 한 번도
읽지 않는다(`test_inner_tune은_검정데이터를_보지_않는다`가 이를 직접 검증한다).
내부 분할도 무작위 K-fold가 아니라 시간순 확장 분할이다 — 무작위를 쓰면 이
연구가 셔플 CV(§7.6, `models.run_leakage_comparison`)에서 비판하는 바로 그
누수를 튜닝 단계에서 한 단계 아래로 옮겨 반복하는 셈이다.

`models.py`는 다른 작업이 동시에 수정 중이라 여기서는 건드리지 않는다.
`models.REG_PARAMS`/`CLF_PARAMS`를 읽어 로컬 딕셔너리로 복사한 뒤 override만
한다 — 원본 모듈은 그대로 둔다.

**세 가지 조건**
- **A (물려받음)**: `models.REG_PARAMS`/`CLF_PARAMS` 그대로, 전처리 없음.
- **B (내부튜닝)**: 폴드별 `inner_tune` 결과로 override, 전처리 없음.
- **C (튜닝+정규화)**: B와 동일하게 폴드별로 튜닝하되(단, 정규화된 피처 위에서
  다시 튜닝한다 — 정규화가 피처의 분포를 바꾸므로 최적 파라미터도 달라질 수
  있다), 날짜별 횡단면 정규화(`cross_sectional_normalize`)를 적용한다.

매 폴드에서 `close_strength` 단독 정렬의 상위10% 수익률도 함께 계산해 모델
대비 상대 우위를 낸다. 정규화는 날짜 내부에서 순서를 보존하는 단조변환이므로
(z-score/rank 모두) `close_strength` 단독 정렬 결과는 조건 A/B/C에서 동일하다
— 폴드당 한 번만 계산해 재사용한다.
"""
from __future__ import annotations

import sys
from itertools import product

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from surge_d1 import config, dataset, models, validate

# --- 그리드 근거 -------------------------------------------------------
# 진단: 현재 CLF_PARAMS는 max_depth=5, learning_rate=0.05, reg_alpha/lambda
# 미지정(=0)이라 학습 AUC가 0.82까지 오른다(외운다). 반대로 REG_PARAMS는
# max_depth=3, learning_rate=0.01, reg_alpha=1.0, reg_lambda=5.0으로
# 과정규화되어 재료 소표본(n~1000)에서 예측이 상수로 붕괴한다. 두 문제가
# 정반대 방향이므로 그리드는 반드시:
#   (1) **깊이를 줄이는 방향**을 포함해야 한다 — max_depth 3도 후보에 넣어
#       현재 분류기의 5가 과도한지 검증한다.
#   (2) **정규화를 추가하는 방향**을 포함해야 한다 — reg_lambda=0(현재
#       분류기 상태)부터 20(과정규화 여부 확인)까지 스캔한다.
#   (3) min_child_samples는 LightGBM에서 트리 잎의 최소 표본수로, 얕은
#       트리·강한 reg_lambda와 별개로 리프 단위 과적합을 억제하는 세 번째
#       축이다. 20(기본값 근처)과 100(재료 소표본 폴드에서도 의미 있게
#       보수적인 값)만 넣어 축을 하나 더 늘리되 조합수는 억제한다.
# learning_rate는 0.02/0.05 두 값만 — n_estimators=200으로 고정돼 있어
# 지나치게 낮추면(예: 0.01) 200트리로 수렴이 안 될 위험이 있고, 지나치게
# 높이면 탐색 범위가 "물려받은 값(0.05)이 최선인지"라는 질문에서 벗어난다.
# 2*2*3*2 = 24 조합 — 스펙이 요구하는 상한(24)에 정확히 맞춘다. 그리드가
# 더 크면 튜닝 자체가 (내부 검증 규모가 작은) 이 문제에서 과적합한다.
PARAM_GRID = {
    "max_depth": [3, 5],
    "learning_rate": [0.02, 0.05],
    "reg_lambda": [0, 5, 20],
    "min_child_samples": [20, 100],
}

_BASE_MODEL_KWARGS = dict(n_estimators=200, subsample=0.8,
                          random_state=config.RANDOM_SEED, verbose=-1,
                          n_jobs=1)   # 재현성 위해 전 조건에서 n_jobs=1 고정


def cross_sectional_normalize(df: pd.DataFrame, feats: list[str],
                              method: str = "rank") -> pd.DataFrame:
    """날짜별(횡단면) 정규화. 스펙: qlib의 CSZScoreNorm과 같은 발상.

    - method="zscore": 그날 평균 0, 분산 1(ddof=0)로 만든다.
    - method="rank": 그날 [0,1] 백분위로 바꾼다(동점 없음, method="first").

    라벨(y_*)·가중치(weight)·식별자(date/code/i 등) 컬럼은 feats에 넣지
    않는 한 절대 건드리지 않는다 — df를 얕은 복사(copy)한 뒤 feats로 지정된
    열만 덮어쓴다.

    NaN 처리: 원본이 NaN인 값은 정규화 후에도 NaN으로 남는다(LightGBM이
    결측을 분기 방향으로 직접 학습하므로 F5 결측 신호를 죽이면 안 된다).
    zscore는 산술 연산(NaN - mean = NaN)으로 자연히 보존되고, rank는
    pandas rank(na_option="keep")로 명시적으로 보존한다.

    하루 표본이 1개(또는 그날 값이 전부 동일)인 날의 zscore는 0/0이라
    정의되지 않는다 — "그 지표가 그날 아무 정보도 주지 못한다"는 뜻으로
    보고 0.0으로 채운다(전체 평균과 같다고 보는 것과 동치). rank도 같은
    경우(그날 유효 표본 1개) (rank-1)/max(n-1,1) = 0/1 = 0.0으로 자연히
    같은 값이 나온다.
    """
    if method not in ("zscore", "rank"):
        raise ValueError(f"알 수 없는 method: {method}")

    out = df.copy()
    grp = out.groupby("date")
    for f in feats:
        if method == "zscore":
            mean = grp[f].transform("mean")
            std = grp[f].transform(lambda s: s.std(ddof=0))
            with np.errstate(invalid="ignore", divide="ignore"):
                z = (out[f] - mean) / std
            # std==0인데 원값이 NaN이 아닌 행만 0.0으로 채운다(위 docstring 참조).
            # 원값이 NaN인 행은 (NaN-mean)/std 연산 자체가 이미 NaN이라 그대로 둔다.
            zero_info = (std == 0) & out[f].notna()
            out[f] = z.mask(zero_info, 0.0)
        else:  # rank
            r = grp[f].rank(method="first", na_option="keep")
            n = grp[f].transform("count")   # 그날 결측 아닌 표본수
            denom = n.where(n > 1, 1)       # n<=1이면 분모를 1로 둬 0/1=0.0
            out[f] = (r - 1) / denom
    return out


def inner_time_splits(dates, n_splits: int = 3
                       ) -> list[tuple[np.ndarray, np.ndarray]]:
    """전달된 프레임 **안에서** 시간순 확장(expanding) 분할을 만든다.

    고유 날짜를 정렬해 (n_splits+1)개의 연속 구간으로 나누고, i번째
    분할은 [0..i) 구간 전체를 학습, i번째 구간을 검증으로 쓴다 — walk-forward
    와 같은 사상(思想)을 내부 튜닝에도 그대로 적용한다. 무작위 K-fold를
    쓰면 미래 날짜가 학습에 섞여 들어가, 이 연구가 §7.6에서 비판하는 셔플
    CV 누수를 튜닝 레벨에서 반복하게 된다.

    반환: [(train_idx, val_idx), ...] — dates(0-base 위치) 기준 정수 인덱스.
    """
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    uniq = np.sort(d.unique())
    chunks = np.array_split(uniq, n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_dates = np.concatenate(chunks[:i])
        val_dates = chunks[i]
        if len(val_dates) == 0 or len(train_dates) == 0:
            continue
        tr_idx = np.where(d.isin(train_dates).to_numpy())[0]
        va_idx = np.where(d.isin(val_dates).to_numpy())[0]
        splits.append((tr_idx, va_idx))
    return splits


def inner_tune(train_df: pd.DataFrame, feats: list[str], grid: dict,
               n_splits: int = 3) -> dict:
    """학습 프레임 내부 시간순 분할의 평균 AUC로 그리드 최적점을 고른다.

    **`train_df` 밖의 어떤 행도 읽지 않는다** — 인자로 받은 프레임과 그
    안에서 만든 `inner_time_splits`만 사용한다(검정 데이터 무열람 원칙).

    분류기(LGBMClassifier) 기준으로 튜닝하고, 회귀는 같은 max_depth/
    learning_rate/reg_lambda/min_child_samples를 그대로 공유한다(호출부인
    run_sensitivity가 반환된 dict를 REG_PARAMS·CLF_PARAMS 양쪽에 동일하게
    override한다) — IC는 표본이 작은 내부 분할에서 상수 붕괴로 정의되지
    않는 경우가 잦아 튜닝 기준으로 불안정하고, AUC는 분류기·회귀기 둘 다
    잘못된 방향(과적합 vs 과소적합)을 같은 축(깊이·정규화)으로 교정하므로
    분류기 AUC 하나로 고른 값을 공유해도 무방하다.
    """
    from lightgbm import LGBMClassifier

    splits = inner_time_splits(train_df["date"], n_splits=n_splits)
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))

    best_params = dict(zip(keys, combos[0]))  # 폴백: 첫 조합
    best_score = -np.inf
    for combo in combos:
        params = dict(zip(keys, combo))
        aucs = []
        for tr_idx, va_idx in splits:
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]
            ytr = tr["y_cls"].to_numpy(dtype=int)
            yva = va["y_cls"].to_numpy(dtype=int)
            if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
                continue
            kwargs = dict(_BASE_MODEL_KWARGS)
            kwargs.update(params)
            clf = LGBMClassifier(**kwargs)
            Xtr = tr[feats].to_numpy(dtype=float)
            Xva = va[feats].to_numpy(dtype=float)
            w = tr["weight"].to_numpy(dtype=float) if "weight" in tr.columns else None
            clf.fit(Xtr, ytr, sample_weight=w)
            prob = clf.predict_proba(Xva)[:, 1]
            aucs.append(roc_auc_score(yva, prob))
        if not aucs:
            continue
        score = float(np.mean(aucs))
        if score > best_score:
            best_score = score
            best_params = params
    return best_params


def _fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame,
                 feats: list[str], reg_params: dict, clf_params: dict,
                 label_reg: str = "y_raw") -> tuple[dict, dict]:
    """models._fit_predict_train_test와 같은 계약(학습·검정 예측 동시 반환)을
    쓰되, reg/clf 하이퍼파라미터를 인자로 받아 조건 A/B/C를 같은 코드
    경로로 돌린다. models.py는 수정하지 않는다(동시 편집 중) — 파라미터
    딕셔너리만 이 모듈에서 override해 넘긴다.
    """
    from lightgbm import LGBMClassifier, LGBMRegressor

    Xtr = train_df[feats].to_numpy(dtype=float)
    Xte = test_df[feats].to_numpy(dtype=float)
    w = train_df["weight"].to_numpy(dtype=float)

    reg = LGBMRegressor(**reg_params)
    reg.fit(Xtr, train_df[label_reg].to_numpy(dtype=float), sample_weight=w)
    pred_reg_tr = reg.predict(Xtr)
    pred_reg_te = reg.predict(Xte)

    clf = LGBMClassifier(**clf_params)
    ytr = train_df["y_cls"].to_numpy(dtype=int)
    clf.fit(Xtr, ytr, sample_weight=w)
    prob_raw_tr = clf.predict_proba(Xtr)[:, 1]
    prob_raw_te = clf.predict_proba(Xte)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(prob_raw_tr, ytr, sample_weight=w)
    prob_cal_tr = np.clip(iso.predict(prob_raw_tr), 0.0, 1.0)
    prob_cal_te = np.clip(iso.predict(prob_raw_te), 0.0, 1.0)

    train_out = {"pred_reg": pred_reg_tr, "prob_raw": prob_raw_tr, "prob_cal": prob_cal_tr}
    test_out = {"pred_reg": pred_reg_te, "prob_raw": prob_raw_te, "prob_cal": prob_cal_te}
    return train_out, test_out


def _resolve_params(overrides: dict | None) -> tuple[dict, dict]:
    """models.REG_PARAMS/CLF_PARAMS를 복사해 override·n_jobs=1을 적용한다."""
    reg_params = dict(models.REG_PARAMS)
    clf_params = dict(models.CLF_PARAMS)
    reg_params["n_jobs"] = 1
    clf_params["n_jobs"] = 1
    if overrides:
        reg_params.update(overrides)
        clf_params.update(overrides)
    return reg_params, clf_params


def run_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """조건 A/B/C × 4폴드(M4 피처셋). 각 행에 close_strength 단독 상위10%
    수익률과 그 대비 상대 우위를 함께 기록한다.
    """
    feats = dataset.FEATURE_SETS["M4"]
    folds = validate.walk_forward_folds(df["date"], models.TEST_YEARS)
    rows = []

    for fi, (tr_mask, te_mask) in enumerate(folds, 1):
        tr_raw = df[tr_mask].reset_index(drop=True)
        te_raw = df[te_mask].reset_index(drop=True)
        # Step 1: 시간순 확인 — 모든 폴드에서 명시적으로 검증한다.
        models.assert_train_before_test(tr_raw, te_raw)
        year = int(te_raw["date"].dt.year.mode()[0])

        # close_strength 단독 정렬은 날짜 내부 순위를 바꾸지 않는 단조
        # 변환(zscore/rank 모두)이므로 조건 A/B/C에서 결과가 동일하다 —
        # 폴드당 한 번만 계산해 재사용한다.
        cs_top = validate.top_decile_return(
            te_raw["y_raw"].to_numpy(dtype=float),
            te_raw["close_strength"].to_numpy(dtype=float),
            te_raw["date"])

        for cond in ("A", "B", "C"):
            if cond == "A":
                overrides = None
                tr, te = tr_raw, te_raw
            elif cond == "B":
                # Step 2: 그 폴드의 학습 구간만으로 튜닝(검정 데이터 무열람).
                overrides = inner_tune(tr_raw, feats, PARAM_GRID, n_splits=3)
                tr, te = tr_raw, te_raw
            else:  # C
                # Step 3: 정규화는 학습·검정 각각의 날짜 안에서 수행한다 —
                # 전체 기간을 합쳐 통계를 내면 미래정보가 새어 들어간다.
                # cross_sectional_normalize는 date로 groupby하므로 원리상
                # 안전하지만(합쳐서 호출해도 그룹 경계는 날짜 자체다),
                # tr_raw/te_raw를 "따로" 정규화해 이 경계를 코드에서도
                # 명시적으로 드러낸다. 정규화로 피처 분포가 바뀌므로
                # 튜닝도 정규화된 학습 데이터 위에서 다시 한다.
                tr = cross_sectional_normalize(tr_raw, feats, method="zscore")
                te = cross_sectional_normalize(te_raw, feats, method="zscore")
                overrides = inner_tune(tr, feats, PARAM_GRID, n_splits=3)

            reg_params, clf_params = _resolve_params(overrides)
            train_out, test_out = _fit_predict(tr, te, feats, reg_params, clf_params)
            test_m = models.evaluate(te, test_out)
            train_m = models.evaluate(tr, train_out)

            margin = (float("nan") if not np.isfinite(cs_top) or cs_top == 0
                     else (test_m["top_decile"] - cs_top) / abs(cs_top))

            rows.append({
                "condition": cond, "fold": fi, "test_year": year,
                "n_train": len(tr), "n_test": len(te),
                "params": "inherited" if overrides is None else str(overrides),
                "test_auc": test_m["auc"], "train_auc": train_m["auc"],
                "gap_auc": train_m["auc"] - test_m["auc"],
                "ic": test_m["ic"], "ic_t": test_m["ic_t"],
                "top_decile": test_m["top_decile"],
                "cs_top_decile": cs_top,
                "margin_over_cs": margin,
            })
            print(f"  조건{cond} fold{fi}({year}) 완료 "
                 f"test_auc={test_m['auc']:.4f} gap={train_m['auc']-test_m['auc']:+.4f} "
                 f"top10={test_m['top_decile']:+.4%} cs_top10={cs_top:+.4%} "
                 f"margin={margin:+.2%} params={overrides}", flush=True)

    return pd.DataFrame(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_parquet(config.DATASET_PATH)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 하이퍼파라미터·전처리 민감도 (조건 A/B/C × 4폴드, M4) ===")
    res = run_sensitivity(df)
    res.to_csv(config.RESULTS_DIR / "tuning_sensitivity.csv", index=False,
              encoding="utf-8-sig")

    print("\n=== 조건별 요약(폴드 평균) ===")
    summary = res.groupby("condition")[
        ["test_auc", "train_auc", "gap_auc", "ic", "top_decile",
         "cs_top_decile", "margin_over_cs"]].mean()
    print(summary.to_string())

    print("\n=== 전체 표 ===")
    print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
