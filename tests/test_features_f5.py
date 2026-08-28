# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from surge_d1 import features_f5


def inv(dates, foreign, inst=None, indiv=None, tv=None):
    n = len(dates)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "indiv": indiv if indiv is not None else [0.0] * n,
        "foreign": foreign,
        "inst": inst if inst is not None else [0.0] * n,
        "fin_invt": [0.0] * n,
        "insurance": [0.0] * n,
        "trd_value": tv if tv is not None else [100.0] * n,
    })


def test_F5_이름이_6개고_반환키와_일치한다():
    d = inv(["2024-01-02", "2024-01-03"], [10.0, 20.0])
    out = features_f5.compute_f5(d, pd.Timestamp("2024-01-03"))
    assert len(features_f5.F5_NAMES) == 6
    assert set(out.keys()) == set(features_f5.F5_NAMES)


def test_거래대금으로_정규화한다():
    # 외국인 순매수 25, 거래대금 100 → 0.25
    d = inv(["2024-01-02"], [25.0], tv=[100.0])
    out = features_f5.compute_f5(d, pd.Timestamp("2024-01-02"))
    assert out["foreign_net_1d"] == pytest.approx(0.25)


def test_5일_합계는_분자분모_모두_합산한다():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    d = inv(dates, [10.0] * 5, tv=[100.0] * 5)
    out = features_f5.compute_f5(d, pd.Timestamp("2024-01-08"))
    assert out["foreign_net_5d"] == pytest.approx(50.0 / 500.0)


def test_미래_데이터를_쓰지_않는다():
    # 조회일 이후 행이 있어도 무시해야 한다 (누수 방지)
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    d = inv(dates, [10.0, 10.0, 9999.0], tv=[100.0] * 3)
    out = features_f5.compute_f5(d, pd.Timestamp("2024-01-03"))
    assert out["foreign_net_1d"] == pytest.approx(0.10)
    assert out["foreign_net_5d"] == pytest.approx(20.0 / 200.0)


def test_수급이_없으면_0이_아니라_NaN():
    out = features_f5.compute_f5(None, pd.Timestamp("2024-01-03"))
    for k in features_f5.F5_NAMES:
        assert np.isnan(out[k]), k


def test_조회일이_데이터_범위_밖이면_NaN():
    d = inv(["2024-01-02"], [10.0])
    out = features_f5.compute_f5(d, pd.Timestamp("2020-01-02"))
    assert np.isnan(out["foreign_net_1d"])


def test_연속순매수일을_센다():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    d = inv(dates, [-5.0, 3.0, 7.0, 2.0])
    out = features_f5.compute_f5(d, pd.Timestamp("2024-01-05"))
    assert out["foreign_streak"] == 3          # 03,04,05 연속 순매수


def test_연속순매도는_음수로_센다():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    d = inv(dates, [5.0, -3.0, -7.0])
    out = features_f5.compute_f5(d, pd.Timestamp("2024-01-04"))
    assert out["foreign_streak"] == -2


def test_거래대금이_0이면_비율은_NaN():
    d = inv(["2024-01-02"], [10.0], tv=[0.0])
    out = features_f5.compute_f5(d, pd.Timestamp("2024-01-02"))
    assert np.isnan(out["foreign_net_1d"])


def test_미래_극단값이_5일합계를_오염시키지_않는다():
    # 조회일 다음날에 극단값을 심어도 5일 합계 결과가 변하지 않아야 한다(누수 방지)
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    d = inv(dates, [10.0] * 5, tv=[100.0] * 5)
    baseline = features_f5.compute_f5(d, pd.Timestamp("2024-01-05"))

    dates2 = dates + ["2024-01-09"]
    d2 = inv(dates2, [10.0] * 5 + [1e9], tv=[100.0] * 5 + [1.0])
    poisoned = features_f5.compute_f5(d2, pd.Timestamp("2024-01-05"))

    assert poisoned["foreign_net_1d"] == pytest.approx(baseline["foreign_net_1d"])
    assert poisoned["foreign_net_5d"] == pytest.approx(baseline["foreign_net_5d"])
    assert poisoned["inst_net_5d"] == pytest.approx(baseline["inst_net_5d"])
    assert poisoned["indiv_net_5d"] == pytest.approx(baseline["indiv_net_5d"])
    assert poisoned["foreign_streak"] == pytest.approx(baseline["foreign_streak"])
