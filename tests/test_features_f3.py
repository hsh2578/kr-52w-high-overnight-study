# -*- coding: utf-8 -*-
"""F3(qlib Alpha158 계열 20개) 테스트. Task 6 브리프 §Step1 + NaN(거래정지) 보강."""
import numpy as np
import pandas as pd
import pytest

from surge_d1 import features_f3


def bar(o, h, l, c, v=1000.0, n=60):
    """마지막 봉만 지정 값이고 나머지는 평탄한 합성 시계열."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame(
        {"Open": np.full(n, 100.0), "High": np.full(n, 101.0),
         "Low": np.full(n, 99.0), "Close": np.full(n, 100.0),
         "Volume": np.full(n, v)},
        index=idx,
    )
    i = n - 1
    df.loc[df.index[i], ["Open", "High", "Low", "Close", "Volume"]] = [o, h, l, c, v]
    return df, i


def trend(n=60, halt_at=None):
    """종가·거래량이 함께 우상향하는 합성 시계열. halt_at 위치는 거래정지로
    표시한다(Open/High/Low/Volume=NaN, Close는 직전값 유지 — sources.mark_halted
    와 동일한 규칙)."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = np.linspace(100.0, 100.0 + (n - 1), n)
    df = pd.DataFrame(
        {"Open": close - 0.5, "High": close + 1.0, "Low": close - 1.0,
         "Close": close, "Volume": np.linspace(1000.0, 1000.0 + 50 * (n - 1), n)},
        index=idx,
    )
    i = n - 1
    if halt_at is not None:
        prev_close = df["Close"].iloc[halt_at - 1] if halt_at > 0 else df["Close"].iloc[0]
        df.loc[df.index[halt_at], ["Open", "High", "Low", "Volume"]] = np.nan
        df.loc[df.index[halt_at], "Close"] = prev_close
    return df, i


def test_F3_이름이_20개고_반환키와_일치한다():
    df, i = bar(100, 110, 90, 105)
    out = features_f3.compute_f3(df, i)
    assert len(features_f3.F3_NAMES) == 20
    assert set(out.keys()) == set(features_f3.F3_NAMES)


def test_KMID_몸통비율():
    df, i = bar(100, 110, 90, 105)
    out = features_f3.compute_f3(df, i)
    assert out["KMID"] == pytest.approx((105 - 100) / 100)


def test_KLEN_고저폭():
    df, i = bar(100, 110, 90, 105)
    out = features_f3.compute_f3(df, i)
    assert out["KLEN"] == pytest.approx((110 - 90) / 100)


def test_KMID2_몸통을_고저폭으로_나눈다():
    df, i = bar(100, 110, 90, 105)
    out = features_f3.compute_f3(df, i)
    assert out["KMID2"] == pytest.approx(5 / 20)


def test_KUP_윗꼬리는_시가종가중_큰값_기준():
    df, i = bar(100, 110, 90, 105)
    out = features_f3.compute_f3(df, i)
    assert out["KUP"] == pytest.approx((110 - 105) / 100)
    assert out["KUP2"] == pytest.approx(5 / 20)


def test_KLOW_아랫꼬리는_시가종가중_작은값_기준():
    df, i = bar(100, 110, 90, 105)
    out = features_f3.compute_f3(df, i)
    assert out["KLOW"] == pytest.approx((100 - 90) / 100)
    assert out["KLOW2"] == pytest.approx(10 / 20)


def test_KSFT_종가편향():
    df, i = bar(100, 110, 90, 105)
    out = features_f3.compute_f3(df, i)
    assert out["KSFT"] == pytest.approx((2 * 105 - 110 - 90) / 100)
    assert out["KSFT2"] == pytest.approx((2 * 105 - 110 - 90) / 20)


def test_고저가_같으면_0으로_나누지_않는다():
    df, i = bar(100, 100, 100, 100)
    out = features_f3.compute_f3(df, i)
    for k, v in out.items():
        assert np.isfinite(v), k


def test_RSV_20_스토캐스틱_위치():
    # 최근 20봉 저가 99, 고가 101 고정 → 종가 101이면 RSV=1
    df, i = bar(100, 101, 99, 101)
    out = features_f3.compute_f3(df, i)
    assert out["RSV_20"] == pytest.approx(1.0)


def test_CNTP_20_상승일비율은_0과_1_사이():
    df, i = bar(100, 110, 90, 105)
    out = features_f3.compute_f3(df, i)
    assert 0.0 <= out["CNTP_20"] <= 1.0


# --- NaN(거래정지 봉)이 20봉 창 안에 섞인 경우 ------------------------------
# sources.mark_halted 규칙: 정지봉은 Open/High/Low/Volume=NaN, Close=직전값
# 유지. 이벤트 당일(i)은 events.is_breakout이 배제하므로 항상 유효하지만,
# 창의 다른 위치(i-19..i-1)에는 정지봉이 얼마든지 섞일 수 있다.

def test_RSV_20_창_안의_NaN은_최대최소_계산에서_제외된다():
    # 평탄한 저가99/고가101 구간 + 마지막 봉만 (99,101,101) → RSV=1 이 되도록
    # 설계한 bar() 시나리오에, 극단값이 아닌 중간 봉 하나를 정지 처리해도
    # 결과가 그대로 1.0 이어야 한다(정지봉의 NaN이 min/max를 오염시키지 않음).
    df, i = bar(100, 101, 99, 101)
    halt_i = i - 5
    df.loc[df.index[halt_i], ["Open", "High", "Low", "Volume"]] = np.nan
    out = features_f3.compute_f3(df, i)
    assert np.isfinite(out["RSV_20"])
    assert out["RSV_20"] == pytest.approx(1.0)


def test_IMAX_20_NaN_고가는_최댓값_후보에서_제외된다():
    df, i = trend(n=60)
    # 창(41..59) 안에서 진짜 최고가 위치를 41(가장 이른 봉)로 만든다.
    df.loc[df.index[41], "High"] = 500.0
    out_before = features_f3.compute_f3(df, i)
    expected_imax = (i - 41) / features_f3.WINDOW
    assert out_before["IMAX_20"] == pytest.approx(expected_imax)

    # 그 봉을 거래정지로 만들어 High를 NaN으로 바꾸면, 최고가 후보에서
    # 제외되고 그다음으로 높은 봉(정상적인 우상향 흐름의 마지막 봉, i)이
    # 선택되어야 한다 — NaN 때문에 index가 틀어지거나 결과가 NaN이 되면 안 된다.
    df.loc[df.index[41], ["Open", "High", "Low", "Volume"]] = np.nan
    df.loc[df.index[41], "Close"] = df["Close"].iloc[40]
    out_after = features_f3.compute_f3(df, i)
    assert np.isfinite(out_after["IMAX_20"])
    assert out_after["IMAX_20"] == pytest.approx(0.0)  # 최고가가 i 자신


def test_CORR_20_거래량_NaN이_있어도_유효표본으로_계산된다():
    df, i = trend(n=60, halt_at=45)
    out = features_f3.compute_f3(df, i)
    # 종가·거래량이 함께 우상향하므로(정지봉 제외) 강한 양의 상관이어야 한다
    assert np.isfinite(out["CORR_20"])
    assert out["CORR_20"] > 0.5


def test_모든_롤링_피처가_창_안의_NaN에도_유한하다():
    df, i = trend(n=60, halt_at=48)
    out = features_f3.compute_f3(df, i)
    for name in features_f3.ROLL_NAMES:
        assert np.isfinite(out[name]), name


def test_모든_롤링_피처가_창_안의_여러_NaN에도_유한하다():
    df, i = trend(n=60)
    for pos in (41, 45, 50, 55):
        df.loc[df.index[pos], ["Open", "High", "Low", "Volume"]] = np.nan
        df.loc[df.index[pos], "Close"] = df["Close"].iloc[pos - 1]
    out = features_f3.compute_f3(df, i)
    for name in features_f3.F3_NAMES:
        assert np.isfinite(out[name]), name


def test_VSTD_20_거래량_NaN이_있어도_유한하다():
    df, i = trend(n=60, halt_at=50)
    out = features_f3.compute_f3(df, i)
    assert np.isfinite(out["VSTD_20"])
    assert out["VSTD_20"] >= 0.0


def test_WVMA_20_거래량_NaN이_있어도_유한하다():
    df, i = trend(n=60, halt_at=50)
    out = features_f3.compute_f3(df, i)
    assert np.isfinite(out["WVMA_20"])
    assert out["WVMA_20"] >= 0.0
