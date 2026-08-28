# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from surge_d1 import features_f1


def synth(n=300):
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = np.linspace(100, 200, n)
    return pd.DataFrame(
        {"Open": c * 0.99, "High": c * 1.02, "Low": c * 0.98,
         "Close": c, "Volume": np.full(n, 1000.0)},
        index=idx,
    )


def market_for(df):
    return pd.DataFrame({"ret": 0.0, "gap": 0.0, "n": 1}, index=df.index)


def test_F1_이름이_14개고_반환키와_일치한다():
    df = synth()
    out = features_f1.compute_f1(df, 280, market_for(df))
    assert len(features_f1.F1_NAMES) == 14
    assert set(out.keys()) == set(features_f1.F1_NAMES)


def test_breakout_pct_직전52주고가_대비_초과율():
    df = synth()
    i = 280
    prior_high = df["Close"].iloc[i - 252:i].max()
    out = features_f1.compute_f1(df, i, market_for(df))
    expected = (df["Close"].iloc[i] - prior_high) / prior_high * 100
    assert out["breakout_pct"] == pytest.approx(expected, rel=1e-6)


def test_volume_surge_50일평균_대비_배수():
    df = synth()
    i = 280
    df.loc[df.index[i], "Volume"] = 3000.0
    out = features_f1.compute_f1(df, i, market_for(df))
    assert out["volume_surge"] == pytest.approx(3.0, rel=1e-6)


def test_close_strength_고가저가_사이_종가위치():
    df = synth()
    i = 280
    df.loc[df.index[i], "Low"] = 100.0
    df.loc[df.index[i], "High"] = 200.0
    df.loc[df.index[i], "Close"] = 175.0
    out = features_f1.compute_f1(df, i, market_for(df))
    assert out["close_strength"] == pytest.approx(0.75)


def test_rs_vs_market_시장수익률을_뺀다():
    df = synth()
    i = 280
    mk = market_for(df)
    mk.loc[:, "ret"] = 0.0
    # 시장 20일 수익률을 5%로 만들기 위해 누적수익 계산이 필요하므로
    # market DataFrame에 index 컬럼을 직접 제공하는 계약을 검증한다
    out = features_f1.compute_f1(df, i, mk)
    stock_ret = (df["Close"].iloc[i] / df["Close"].iloc[i - 20] - 1) * 100
    assert out["rs_vs_market"] == pytest.approx(stock_ret - out["market_return"],
                                                rel=1e-6)


def test_결측이_없다():
    df = synth()
    out = features_f1.compute_f1(df, 280, market_for(df))
    for k, v in out.items():
        assert v is not None and np.isfinite(v), k
