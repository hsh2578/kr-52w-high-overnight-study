# -*- coding: utf-8 -*-
"""검증 프로토콜: walk-forward + embargo, 지표, 통계검정. 스펙 §7.2, §7.5."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from surge_d1 import config


def walk_forward_folds(dates: pd.Series, test_years: list[int],
                       embargo_days: int = config.EMBARGO_DAYS
                       ) -> list[tuple[np.ndarray, np.ndarray]]:
    """expanding window walk-forward. 반환: [(train_mask, test_mask), ...]

    학습은 검정연도 시작 이전 전체에서 embargo_days(달력일) 만큼을 더 잘라낸다.
    마스크는 입력 dates의 위치(순서)에 정렬된 불린 numpy 배열이다.
    """
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    folds = []
    for y in sorted(test_years):
        start = pd.Timestamp(f"{y}-01-01")
        end = pd.Timestamp(f"{y}-12-31 23:59:59")
        test = (d >= start) & (d <= end)
        cutoff = start - pd.Timedelta(days=embargo_days)
        train = d < cutoff
        if train.sum() == 0 or test.sum() == 0:
            continue
        folds.append((train.to_numpy(), test.to_numpy()))
    return folds


def spearman_ic(y_true, y_pred, dates) -> tuple[float, float]:
    """일별 Spearman 순위상관의 평균과 t값(=IR 근사).

    주의: 이는 풀링된(전체를 한 번에 상관시킨) IC가 아니라, 날짜별로 상관을
    각각 계산한 뒤 그 시계열의 평균/표준오차로 t값을 낸 것이다(일별 IC 평균).
    """
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float),
                       "p": np.asarray(y_pred, dtype=float),
                       "d": pd.to_datetime(pd.Series(dates).values)})
    ics = []
    for _, g in df.groupby("d"):
        if len(g) < 3 or g["p"].nunique() < 2 or g["y"].nunique() < 2:
            continue
        ic = stats.spearmanr(g["y"], g["p"]).statistic
        if np.isfinite(ic):
            ics.append(float(ic))
    if not ics:
        return 0.0, 0.0
    arr = np.array(ics)
    se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.nan
    t = float(arr.mean() / se) if se and np.isfinite(se) and se > 0 else 0.0
    return float(arr.mean()), t


def top_decile_return(y_true, y_pred, dates, q: float = 0.1) -> float:
    """일별로 예측 상위 q 비율을 뽑아 실제 수익률 평균을 낸다.

    예측값에 변동이 없는 날(전부 동률)은 상위 선정이 무의미하므로 건너뛴다.

    최소 표본 가드: 이름이 3개 미만인 날은 제외한다(spearman_ic의
    len(g) < 3과 동일 기준 — 절대 개수 3을 무비판적으로 베낀 것이 아니라
    같은 통계적 근거로 다시 도출한 값이다). 근거: q=0.1과 반올림
    (k = max(1, round(n*q)))의 조합상, n이 1~14 사이인 날은 전부 선정
    개수 k가 1로 고정된다. n=1인 날은 후보가 하나뿐이라 예측이 무엇이든
    무조건 그 종목이 뽑히는 자명한 경우이고, n=2인 날은 사실상 동전
    던지기 수준의 이산적 비교(둘 중 하나)에 불과해 개별 표본의 노이즈가
    극단적으로 크다. n>=3부터는 최소한 "상위 1/3 이내"라는 의미 있는
    배제가 성립한다.

    참고: `nunique < 2` 조건은 이름이 1개뿐인 날을 우연히 걸러내지만
    (값이 하나면 정의상 nunique==1) 이름이 2개뿐인 날(서로 다른 예측값
    2개 → nunique==2)은 걸러내지 못한다 — 그래서 표본 크기 자체를 보는
    `len(g) < 3` 가드가 별도로 필요하다.

    실측(data/dataset.parquet, 1,586개 이벤트일): n=1인 날 53개(3.3%),
    n=2인 날 63개(4.0%), 합쳐서 116개(7.3%)가 이 가드로 제외된다.
    """
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float),
                       "p": np.asarray(y_pred, dtype=float),
                       "d": pd.to_datetime(pd.Series(dates).values)})
    picked = []
    for _, g in df.groupby("d"):
        if len(g) < 3 or g["p"].nunique() < 2:
            continue
        k = max(1, int(round(len(g) * q)))
        picked.append(g.nlargest(k, "p")["y"])
    if not picked:
        return float("nan")
    return float(pd.concat(picked).mean())


def bootstrap_ci(values, n: int = 1000, seed: int = config.RANDOM_SEED,
                 alpha: float = 0.05) -> tuple[float, float]:
    """평균의 부트스트랩 신뢰구간. 동일 seed면 항상 동일한 결과를 낸다."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def permutation_pvalue(metric_fn, y, pred_a, pred_b, dates,
                       n: int = 1000, seed: int = config.RANDOM_SEED) -> float:
    """H0: pred_b가 pred_a보다 낫지 않다.

    관측 차이(metric(b) - metric(a))가, b의 예측을 무작위 섞었을 때의 분포에서
    얼마나 극단적인지로 p값을 낸다. (count+1)/(n+1) 보정으로 정확히 0을
    반환하지 않는다. 동일 seed면 항상 동일한 결과를 낸다.
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    observed = metric_fn(y, b, dates) - metric_fn(y, a, dates)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        shuffled = rng.permutation(b)
        diff = metric_fn(y, shuffled, dates) - metric_fn(y, a, dates)
        if diff >= observed:
            cnt += 1
    return (cnt + 1) / (n + 1)


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH 보정 p값. 입력 순서를 유지해 반환한다."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out.tolist()
