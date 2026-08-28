# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from surge_d1 import validate


def dates_for(years):
    out = []
    for y in years:
        out += [pd.Timestamp(f"{y}-03-01"), pd.Timestamp(f"{y}-09-01")]
    return pd.Series(pd.to_datetime(out))


def test_walk_forward_학습이_검정보다_과거다():
    d = dates_for([2020, 2021, 2022, 2023, 2024])
    folds = validate.walk_forward_folds(d, test_years=[2023, 2024],
                                        embargo_days=0)
    assert len(folds) == 2
    for tr, te in folds:
        assert d[tr].max() < d[te].min()


def test_walk_forward_expanding_이다():
    d = dates_for([2020, 2021, 2022, 2023, 2024])
    folds = validate.walk_forward_folds(d, test_years=[2023, 2024],
                                        embargo_days=0)
    assert folds[1][0].sum() > folds[0][0].sum()


def test_embargo가_학습_끝부분을_잘라낸다():
    d = pd.Series(pd.to_datetime(
        ["2022-12-20", "2022-12-28", "2023-06-01"]))
    no_emb = validate.walk_forward_folds(d, [2023], embargo_days=0)[0][0].sum()
    with_emb = validate.walk_forward_folds(d, [2023], embargo_days=5)[0][0].sum()
    assert with_emb < no_emb


def test_walk_forward_마스크는_불린_배열이고_길이가_입력과_같다():
    d = dates_for([2020, 2021, 2022, 2023, 2024])
    folds = validate.walk_forward_folds(d, test_years=[2023, 2024],
                                        embargo_days=0)
    for tr, te in folds:
        assert tr.dtype == bool
        assert te.dtype == bool
        assert len(tr) == len(d)
        assert len(te) == len(d)
        assert not np.any(tr & te)


def test_spearman_ic_완전상관이면_1에_가깝다():
    d = pd.Series(pd.to_datetime(["2024-01-02"] * 5 + ["2024-01-03"] * 5))
    y = np.array([1, 2, 3, 4, 5] * 2, dtype=float)
    ic, t = validate.spearman_ic(y, y, d)
    assert ic == pytest.approx(1.0)


def test_top_decile_return_상위픽만_평균낸다():
    d = pd.Series(pd.to_datetime(["2024-01-02"] * 10))
    y = np.arange(10, dtype=float)
    pred = np.arange(10, dtype=float)
    # 상위 10% = 1개 = y가 가장 큰 9.0
    assert validate.top_decile_return(y, pred, d, q=0.1) == pytest.approx(9.0)


def test_bootstrap_ci가_평균을_감싼다():
    rng = np.random.default_rng(0)
    v = rng.normal(0.5, 1.0, 500)
    lo, hi = validate.bootstrap_ci(v, n=200, seed=42)
    assert lo < v.mean() < hi


def test_bootstrap_ci_결정적이다():
    rng = np.random.default_rng(0)
    v = rng.normal(0.5, 1.0, 500)
    a = validate.bootstrap_ci(v, n=200, seed=42)
    b = validate.bootstrap_ci(v, n=200, seed=42)
    assert a == b


def test_permutation_pvalue_0을_반환하지_않는다():
    rng = np.random.default_rng(1)
    d = pd.Series(pd.to_datetime(["2024-01-02"] * 20))
    y = rng.normal(size=20)
    a = rng.normal(size=20)
    # b는 y와 완전히 동일한 순위를 갖도록 만들어 관측 차이를 극단적으로 크게 만든다
    b = y.copy()

    def ic_fn(yy, pp, dd):
        ic, _ = validate.spearman_ic(yy, pp, dd)
        return ic

    p = validate.permutation_pvalue(ic_fn, y, a, b, d, n=200, seed=42)
    assert p > 0.0


def test_permutation_pvalue_결정적이다():
    rng = np.random.default_rng(1)
    d = pd.Series(pd.to_datetime(["2024-01-02"] * 20))
    y = rng.normal(size=20)
    a = rng.normal(size=20)
    b = rng.normal(size=20)

    def ic_fn(yy, pp, dd):
        ic, _ = validate.spearman_ic(yy, pp, dd)
        return ic

    p1 = validate.permutation_pvalue(ic_fn, y, a, b, d, n=200, seed=42)
    p2 = validate.permutation_pvalue(ic_fn, y, a, b, d, n=200, seed=42)
    assert p1 == p2


def test_benjamini_hochberg_단조증가하고_원래값_이상이다():
    p = [0.001, 0.01, 0.03, 0.5]
    adj = validate.benjamini_hochberg(p)
    assert all(a >= b - 1e-12 for a, b in zip(adj, p))
    assert adj == sorted(adj)


def test_benjamini_hochberg_입력_순서를_유지한다():
    p = [0.5, 0.001, 0.03, 0.01]
    adj = validate.benjamini_hochberg(p)
    # 가장 작은 p값(index 1)의 보정값이 가장 작아야 한다
    assert adj[1] == min(adj)
    assert len(adj) == len(p)


def test_benjamini_hochberg_모두_1이하():
    adj = validate.benjamini_hochberg([0.9, 0.95, 0.99])
    assert all(0.0 <= a <= 1.0 for a in adj)
