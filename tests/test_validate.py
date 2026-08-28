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


def test_spearman_ic_전역상관이_아니라_일별평균이다():
    # 하루씩 보면(날짜별 3개) 완전한 순위상관(+1)이지만, 날짜 간 수준이
    # 뒤바뀌어 있어(1일차: 1,2,3 / 2일차: -3,-2,-1) 전체를 풀링해 한 번에
    # spearmanr을 내면 -0.5429가 나온다(직접 계산 확인). 이 값이 아니라
    # +1.0이 나와야 "일별 계산 후 평균" 구현이 맞다 — 잘못 풀링하는
    # 구현으로 회귀하면 이 테스트가 실패한다.
    d = pd.Series(pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3))
    y = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    pred = np.array([1, 2, 3, -3, -2, -1], dtype=float)
    ic, _ = validate.spearman_ic(y, pred, d)
    assert ic == pytest.approx(1.0)


def test_top_decile_return_상위픽만_평균낸다():
    d = pd.Series(pd.to_datetime(["2024-01-02"] * 10))
    y = np.arange(10, dtype=float)
    pred = np.arange(10, dtype=float)
    # 상위 10% = 1개 = y가 가장 큰 9.0
    assert validate.top_decile_return(y, pred, d, q=0.1) == pytest.approx(9.0)


def test_top_decile_return_일별선정이지_전역컷오프가_아니다():
    # 전역(풀링) 상위 10%로 잘못 구현하면, 예측값이 절대적으로 큰 1일차
    # (91~100)가 픽을 독점해 상위 2개(pred=100,99 → y=-10,-9)의 평균인
    # -9.5가 나온다. 올바른 "일별" 구현은 각 날짜에서 따로 10%(=10개 중
    # 1개)씩 뽑으므로 1일차에서 pred=100(y=-10), 2일차에서 pred=10(y=10)이
    # 뽑혀 평균 0.0이 된다. 두 값이 부호까지 다를 만큼 크게 벌어지므로
    # 전역 컷오프로 퇴화하면 이 테스트가 확실히 실패한다.
    d = pd.Series(pd.to_datetime(["2024-01-02"] * 10 + ["2024-01-03"] * 10))
    y = np.array(list(range(-10, 0)) + list(range(1, 11)), dtype=float)
    pred = np.array(list(range(100, 90, -1)) + list(range(1, 11)), dtype=float)
    assert validate.top_decile_return(y, pred, d, q=0.1) == pytest.approx(0.0)


def test_top_decile_return_표본이_너무_작은_날은_제외한다():
    # 회귀 방지: 이전 버전은 `len(g) < 1`이 죽은 코드였다(groupby는 빈
    # 그룹을 만들지 않으므로 결코 True가 될 수 없다) — 실질적 가드는
    # `nunique < 2` 하나뿐이었다. 이 가드는 이름이 1개뿐인 날은 우연히
    # 걸러내지만(값이 하나면 정의상 nunique==1) 이름이 2개뿐인 날(서로
    # 다른 예측값 2개 → nunique==2)은 걸러내지 못해, 사실상 동전 던지기
    # 수준의 극단적 노이즈가 그대로 평균에 섞였다.
    # 아래는 이름 2개짜리 날(극단값 y=100 포함)과 이름 3개짜리 정상적인
    # 날을 섞었다. 옛 코드(len(g)<1만 죽은 채 nunique<2만 남은 버전)는
    # 2개짜리 날의 y=100을 포함해 평균 51.5를 내지만, 최소 3개 표본
    # 가드가 있는 올바른 구현은 2개짜리 날을 통째로 제외하고 3개짜리
    # 날에서만 뽑아 평균 3.0을 낸다.
    d = pd.Series(pd.to_datetime(
        ["2024-01-02"] * 2 + ["2024-01-03"] * 3))
    y = np.array([100.0, 1.0, 1.0, 2.0, 3.0])
    pred = np.array([2.0, 1.0, 1.0, 2.0, 3.0])
    assert validate.top_decile_return(y, pred, d, q=0.1) == pytest.approx(3.0)


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
