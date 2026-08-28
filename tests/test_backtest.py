# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from surge_d1 import backtest


def toy():
    """2일 × 4종목. score가 높을수록 실제 수익률도 높게 만들어 둔다."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 4 + ["2024-01-03"] * 4),
        "code": list("abcd") * 2,
        "y_raw": [0.04, 0.02, -0.01, -0.03] * 2,
        "score": [0.9, 0.7, 0.3, 0.1] * 2,
    })


def test_simulate_상위N만_균등매수한다():
    d = backtest.simulate(toy(), "score", top_n=2)
    # 상위 2개 = 0.04, 0.02 → 평균 0.03
    assert d.loc[pd.Timestamp("2024-01-02"), "ret"] == pytest.approx(0.03)
    assert d.loc[pd.Timestamp("2024-01-02"), "n_picked"] == 2


def test_simulate_후보가_N보다_적으면_있는것만_산다():
    df = toy()
    df = df[df["code"].isin(["a", "b"])]
    d = backtest.simulate(df, "score", top_n=5)
    assert d.iloc[0]["n_picked"] == 2
    assert d.iloc[0]["ret"] == pytest.approx(0.03)


def test_simulate_전부매수는_그날_평균이다():
    d = backtest.simulate(toy(), "score", top_n=None)
    assert d.iloc[0]["ret"] == pytest.approx((0.04 + 0.02 - 0.01 - 0.03) / 4)
    assert d.iloc[0]["n_picked"] == 4


def test_performance_stats_누적수익률은_복리다():
    daily = pd.DataFrame(
        {"ret": [0.10, 0.10], "n_picked": [1, 1]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    s = backtest.performance_stats(daily)
    assert s["cum_return"] == pytest.approx(0.21)      # 1.1*1.1 - 1
    assert s["win_rate"] == pytest.approx(1.0)
    assert s["n_days"] == 2


def test_performance_stats_MDD는_음수가_아니고_낙폭을_잡는다():
    daily = pd.DataFrame(
        {"ret": [0.5, -0.5, 0.0], "n_picked": [1, 1, 1]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    s = backtest.performance_stats(daily)
    # 1.5 → 0.75, 고점 1.5 대비 낙폭 0.5
    assert s["mdd"] == pytest.approx(0.5)
    assert s["mdd"] >= 0


def test_performance_stats_sortino는_하방편차만_벌한다():
    # 상승만 있는 계열은 하방편차 0 → Sortino가 NaN(정의 불가)
    up = pd.DataFrame({"ret": [0.01] * 40, "n_picked": [1] * 40},
                      index=pd.bdate_range("2024-01-01", periods=40))
    assert np.isnan(backtest.performance_stats(up)["sortino"])
    # 하락이 섞이면 유한한 값이 나온다
    mixed = up.copy()
    mixed.iloc[0, mixed.columns.get_loc("ret")] = -0.02
    assert np.isfinite(backtest.performance_stats(mixed)["sortino"])


def test_performance_stats_profit_factor는_총이익_대_총손실():
    daily = pd.DataFrame(
        {"ret": [0.03, -0.01, 0.01], "n_picked": [1, 1, 1]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    s = backtest.performance_stats(daily)
    assert s["profit_factor"] == pytest.approx(0.04 / 0.01)
    assert s["expectancy"] == pytest.approx(0.01)


def test_performance_stats_SQN은_표본_30미만이면_NaN():
    small = pd.DataFrame({"ret": [0.01, -0.01] * 5, "n_picked": [1] * 10},
                         index=pd.bdate_range("2024-01-01", periods=10))
    assert np.isnan(backtest.performance_stats(small)["sqn"])
    big = pd.DataFrame({"ret": [0.01, -0.005] * 20, "n_picked": [1] * 40},
                       index=pd.bdate_range("2024-01-01", periods=40))
    assert np.isfinite(backtest.performance_stats(big)["sqn"])


def test_performance_stats_mdd_days는_낙폭_지속_거래일수():
    # 고점 후 3일 연속 고점 미만 → mdd_days = 3
    daily = pd.DataFrame(
        {"ret": [0.10, -0.05, -0.01, 0.02, 0.20], "n_picked": [1] * 5},
        index=pd.bdate_range("2024-01-01", periods=5),
    )
    assert backtest.performance_stats(daily)["mdd_days"] == 3


def test_performance_stats_beta는_벤치마크와_동일하면_1():
    idx = pd.bdate_range("2024-01-01", periods=40)
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0, 0.01, 40), index=idx)
    daily = pd.DataFrame({"ret": b.to_numpy(), "n_picked": [1] * 40}, index=idx)
    s = backtest.performance_stats(daily, benchmark=b)
    assert s["beta"] == pytest.approx(1.0, abs=1e-9)


def test_performance_stats_빈_입력도_죽지_않는다():
    daily = pd.DataFrame({"ret": [], "n_picked": []},
                         index=pd.to_datetime([]))
    s = backtest.performance_stats(daily)
    assert s["n_days"] == 0
    assert np.isnan(s["sharpe"]) or s["sharpe"] == 0.0


def test_random_baseline_시드고정이면_재현된다():
    a = backtest.simulate_random(toy(), top_n=2, n_iter=5, seed=42)
    b = backtest.simulate_random(toy(), top_n=2, n_iter=5, seed=42)
    pd.testing.assert_frame_equal(a, b)


def _normal_daily(n=250, mu=0.001, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n),
                     index=pd.bdate_range("2020-01-01", periods=n))


def test_dsr_시행이_1개면_평범한_PSR과_같다():
    r = _normal_daily()
    out = backtest.deflated_sharpe_ratio(r, n_trials=1)
    assert out["sr0"] == pytest.approx(0.0)
    assert 0.0 <= out["dsr"] <= 1.0
    # SR* = 0 대비 확률적 샤프비(PSR)와 정확히 같아야 한다
    from scipy import stats as sps
    sr_hat = out["sr_hat"]
    denom = np.sqrt(1 - out["skew"] * sr_hat
                    + ((out["kurtosis"] - 1) / 4) * sr_hat ** 2)
    expected = sps.norm.cdf(sr_hat * np.sqrt(len(r) - 1) / denom)
    assert out["dsr"] == pytest.approx(expected)


def test_dsr_시행수가_늘면_DSR이_낮아진다():
    r = _normal_daily(n=500, mu=0.0015, sigma=0.01)
    trial_sr = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    few = backtest.deflated_sharpe_ratio(r, n_trials=2, trial_sharpes=trial_sr)
    many = backtest.deflated_sharpe_ratio(r, n_trials=200, trial_sharpes=trial_sr)
    assert many["sr0"] > few["sr0"]
    assert many["dsr"] <= few["dsr"]


def test_dsr_범위는_0과_1_사이():
    r = _normal_daily(n=300, mu=0.0, sigma=0.02)
    out = backtest.deflated_sharpe_ratio(r, n_trials=11)
    assert 0.0 <= out["dsr"] <= 1.0


def test_strategy_correlation_대각선은_1이고_대칭이다():
    daily = pd.DataFrame({
        "a": [0.01, -0.02, 0.03, 0.0, 0.01],
        "b": [0.02, -0.01, 0.02, 0.01, -0.01],
    }, index=pd.bdate_range("2024-01-01", periods=5))
    corr = backtest.strategy_correlation(daily)
    assert corr.loc["a", "a"] == pytest.approx(1.0)
    assert corr.loc["a", "b"] == pytest.approx(corr.loc["b", "a"])
    assert (corr.to_numpy() <= 1.0 + 1e-9).all()
    assert (corr.to_numpy() >= -1.0 - 1e-9).all()
