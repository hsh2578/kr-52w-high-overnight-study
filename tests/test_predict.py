# -*- coding: utf-8 -*-
"""predict.py 단위 테스트.

핵심 함정: events.is_breakout은 i+1<len(df)를 요구해 마지막 봉엔 못 쓴다.
예측 전용 판정(_is_breakout_last/find_today_events)은 그 요구만 뺀 나머지
조건(신고가·거래량·거래대금·i>=260·거래정지·가격제한·제외종목)을 전부
동일하게 적용해야 한다 — 그렇지 않으면 라이브 데모가 연구 스크리닝과
어긋난다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import predict
from surge_d1 import config, dataset


def synth_breakout(n=300, i=None, halted_at=None, extra_move=None):
    """i봉이 52주 신고가 돌파가 되도록 만든 합성 시계열.

    i가 None이면 마지막 봉(n-1)을 돌파봉으로 만든다. i < n-1이면 돌파봉
    뒤에도 정상적인 봉이 이어진다(잘라내기 일관성 테스트용).
    halted_at: 거래정지로 만들 봉의 정수 인덱스(halted 컬럼 True, OHLV NaN).
    extra_move: (prev_close_ratio,) 형태로 i-1봉 종가를 조정해 전일대비
    등락폭을 인위적으로 키울 때 쓴다.
    """
    if i is None:
        i = n - 1
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = np.full(n, 100.0)
    df = pd.DataFrame(
        {"Open": c * 0.99, "High": c * 1.02, "Low": c * 0.98,
         "Close": c.copy(), "Volume": np.full(n, 1000.0),
         "halted": False},
        index=idx,
    )
    if extra_move is not None:
        df.loc[df.index[i - 1], "Close"] = 100.0 * extra_move
    # +28%: 52주 신고가·거래대금 조건은 만족하면서 MAX_DAILY_MOVE(32%) 이내로
    # 유지한다 (기준선 케이스가 정상적으로 돌파 판정을 받으려면 필요).
    df.loc[df.index[i], "Close"] = 128.0
    df.loc[df.index[i], "High"] = 132.0
    df.loc[df.index[i], "Low"] = 118.0
    df.loc[df.index[i], "Volume"] = 1e9
    if halted_at is not None:
        df.loc[df.index[halted_at], "halted"] = True
        df.loc[df.index[halted_at], ["Open", "High", "Low", "Volume"]] = np.nan
    return df


def make_market(df):
    return pd.DataFrame({"ret": 0.0, "gap": 0.0, "n": 1}, index=df.index)


def test_마지막봉_돌파를_잡는다():
    df = synth_breakout()
    ohlcv = {"000001": df}
    market = make_market(df)
    out = predict.find_today_events(ohlcv, market, df.index[-1])
    assert len(out) == 1
    assert out.iloc[0]["code"] == "000001"


def test_돌파가_아니면_비어있다():
    df = synth_breakout()
    df.loc[df.index[-1], "Volume"] = 1000.0   # 거래량 조건 탈락
    ohlcv = {"000001": df}
    market = make_market(df)
    out = predict.find_today_events(ohlcv, market, df.index[-1])
    assert len(out) == 0


def test_피처컬럼이_M4와_일치한다():
    df = synth_breakout()
    ohlcv = {"000001": df}
    market = make_market(df)
    out = predict.find_today_events(ohlcv, market, df.index[-1])
    assert len(out) == 1
    for c in dataset.FEATURE_SETS["M4"]:
        assert c in out.columns, c


def test_제외종목은_후보에서_빠진다(monkeypatch):
    df = synth_breakout()
    ohlcv = {"000001": df}
    market = make_market(df)
    monkeypatch.setattr(predict.events, "load_excluded_codes",
                        lambda: {"000001"})
    out = predict.find_today_events(ohlcv, market, df.index[-1])
    assert len(out) == 0


def test_최근5봉_이내_거래정지가_있으면_제외된다():
    n = 300
    i = n - 1
    df = synth_breakout(n=n, i=i, halted_at=i - 3)   # HALT_LOOKBACK=5 안쪽
    ohlcv = {"000001": df}
    market = make_market(df)
    out = predict.find_today_events(ohlcv, market, df.index[-1])
    assert len(out) == 0


def test_5봉보다_이전_거래정지는_영향없다():
    n = 300
    i = n - 1
    df = synth_breakout(n=n, i=i, halted_at=i - 6)   # HALT_LOOKBACK=5 바깥
    ohlcv = {"000001": df}
    market = make_market(df)
    out = predict.find_today_events(ohlcv, market, df.index[-1])
    assert len(out) == 1


def test_가격제한_초과는_제외된다():
    # 전일종가 90 -> 당일 128: (128/90-1)=42.2% > MAX_DAILY_MOVE(32%)
    df = synth_breakout(extra_move=0.9)
    ohlcv = {"000001": df}
    market = make_market(df)
    out = predict.find_today_events(ohlcv, market, df.index[-1])
    assert len(out) == 0


def test_직전봉으로_잘라도_같은_판정과_피처값():
    n, i = 310, 300
    df_full = synth_breakout(n=n, i=i)
    df_trunc = df_full.iloc[:i + 1].copy()

    market_full = make_market(df_full)
    market_trunc = make_market(df_trunc)

    asof = df_full.index[i]
    out_full = predict.find_today_events({"000001": df_full}, market_full, asof)
    out_trunc = predict.find_today_events({"000001": df_trunc}, market_trunc, asof)

    assert len(out_full) == 1
    assert len(out_trunc) == 1
    for c in dataset.FEATURE_SETS["M4"]:
        v_full = out_full.iloc[0][c]
        v_trunc = out_trunc.iloc[0][c]
        if pd.isna(v_full) and pd.isna(v_trunc):
            continue
        assert v_full == pytest.approx(v_trunc), c


def test_학습데이터는_기준일_이전만_쓴다(monkeypatch, tmp_path):
    rows = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "code": ["000001", "000002", "000003"],
        "weight": [1.0, 1.0, 1.0],
        "y_raw": [0.01, -0.01, 0.02],
        "y_cls": [1, 0, 1],
    })
    path = tmp_path / "dataset.parquet"
    rows.to_parquet(path)
    monkeypatch.setattr(config, "DATASET_PATH", path)

    asof = pd.Timestamp("2026-01-03")
    train = predict.load_training(asof)
    assert len(train) == 2
    assert train["date"].max() < asof


def test_학습데이터에_기준일이후가_섞이면_예외():
    bad = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-05"]),
        "code": ["000001", "000002"],
        "weight": [1.0, 1.0],
        "y_raw": [0.01, -0.01],
        "y_cls": [1, 0],
    })
    with pytest.raises(AssertionError):
        predict._assert_train_before_asof(bad, pd.Timestamp("2026-01-03"))
