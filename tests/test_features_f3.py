# -*- coding: utf-8 -*-
"""F3(qlib Alpha158 계열 20개) 테스트. Task 6 브리프 §Step1 + NaN(거래정지)
보강 + 2026-08-28 리뷰 수정(치명 2건·중요 3건·사소 1건) 회귀 방지."""
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


def synth_from_closes(closes, vols=None, n=None):
    """종가(필요하면 거래량)를 직접 지정한 합성 시계열. 마지막 봉이 오늘(i)이다.
    High=Close+1, Low=Close-1, Open=Close로 둔다(KBAR/RSV/IMAX가 자연스럽게
    유한하도록)."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes) if n is None else n
    idx = pd.bdate_range("2020-01-01", periods=n)
    if vols is None:
        vols = np.full(n, 1000.0)
    else:
        vols = np.asarray(vols, dtype=float)
    df = pd.DataFrame(
        {"Open": closes.copy(), "High": closes + 1.0, "Low": closes - 1.0,
         "Close": closes, "Volume": vols},
        index=idx,
    )
    return df, n - 1


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


# --- 사소7: $open·$close 단일값 분모는 EPS 없이, 무효면 NaN ------------------

def test_KMID_등_시가0이면_NaN을_반환한다():
    # o=0인 비정상 봉(실데이터에서는 발생하지 않지만 가드 자체를 검증한다).
    # EPS로 얼버무려 거대한 유한값을 만드는 대신 NaN을 반환해야 한다.
    df, i = bar(0.0, 5.0, -1.0, 3.0)
    out = features_f3.compute_f3(df, i)
    for k in ("KMID", "KLEN", "KUP", "KLOW", "KSFT"):
        assert np.isnan(out[k]), k
    # "2"-접미사 4종은 (High-Low)+EPS로 나누므로 o=0과 무관하게 유한해야 한다
    for k in ("KMID2", "KUP2", "KLOW2", "KSFT2"):
        assert np.isfinite(out[k]), k


# --- 치명1: IMAX_20 — qlib 1-기반 위치(오래된 봉=1, 오늘=d) -----------------

def test_IMAX_20_오늘이_창_최고가면_1이다():
    df, i = bar(100, 200, 90, 105)  # 오늘 High=200이 창 안 다른 어떤 값보다 크다
    out = features_f3.compute_f3(df, i)
    assert out["IMAX_20"] == pytest.approx(1.0)


def test_IMAX_20_NaN_고가는_최댓값_후보에서_제외된다():
    df, i = trend(n=60)
    # 창(40..59) 안에서 진짜 최고가 위치를 41(창 내 두 번째, 오래된 쪽에서
    # 1-기반 위치 2)로 만든다.
    df.loc[df.index[41], "High"] = 500.0
    out_before = features_f3.compute_f3(df, i)
    expected_imax = ((41 - 40) + 1) / features_f3.WINDOW  # 1-기반 위치 2 / 20
    assert out_before["IMAX_20"] == pytest.approx(expected_imax)

    # 그 봉을 거래정지로 만들어 High를 NaN으로 바꾸면, 최고가 후보에서
    # 제외되고 원래의 우상향 추세대로 오늘(i)이 다시 창의 최고가가 되어야
    # 한다 — NaN 때문에 인덱스가 틀어지거나 결과가 NaN이 되면 안 된다.
    df.loc[df.index[41], ["Open", "High", "Low", "Volume"]] = np.nan
    df.loc[df.index[41], "Close"] = df["Close"].iloc[40]
    out_after = features_f3.compute_f3(df, i)
    assert np.isfinite(out_after["IMAX_20"])
    assert out_after["IMAX_20"] == pytest.approx(1.0)  # 최고가가 오늘(i) 자신


# --- 치명2: VSTD_20 — 분모는 창 평균이 아니라 "오늘"의 거래량 ---------------

def test_VSTD_20_분모는_오늘_거래량이고_창_평균이_아니다():
    n = 60
    vols = np.full(n, 1000.0)
    vols[-1] = 5000.0  # 오늘만 거래량이 5배
    df, i = synth_from_closes(np.full(n, 100.0), vols=vols)
    out = features_f3.compute_f3(df, i)
    vol_window = vols[-20:]
    expected = vol_window.std() / (5000.0 + 1e-12)  # 분모=오늘(5000), 창 평균(1200)이 아님
    assert out["VSTD_20"] == pytest.approx(expected)
    # 창 평균으로 나눈 (틀린) 값과는 뚜렷이 달라야 한다
    wrong = vol_window.std() / (vol_window.mean() + 1e-12)
    assert out["VSTD_20"] != pytest.approx(wrong)


# --- 중요3: WVMA_20 — 분자·분모 모두 절대수익률 -----------------------------

def test_WVMA_20_분자도_절대수익률을_쓴다():
    n = 21
    closes = 100.0 + 5.0 * np.sin(np.arange(n))  # 부호가 자주 바뀌는 지그재그
    vols = 1000.0 + 50.0 * np.arange(n)
    df, i = synth_from_closes(closes, vols=vols)
    out = features_f3.compute_f3(df, i)

    close_e = closes
    vol_e = vols
    price_ret = (close_e[1:] - close_e[:-1]) / close_e[:-1]
    prod_abs = np.abs(price_ret) * vol_e[1:]
    expected = np.std(prod_abs) / (np.mean(prod_abs) + 1e-12)
    assert out["WVMA_20"] == pytest.approx(expected)

    # 분자에 부호를 남겨둔 (틀린) 계산과는 값이 달라야 한다
    prod_signed = price_ret * vol_e[1:]
    wrong = np.std(prod_signed) / (np.mean(np.abs(prod_signed)) + 1e-12)
    assert out["WVMA_20"] != pytest.approx(wrong)


# --- 중요4: CORD_20 — 거래량항은 log(V_t/V_{t-1} + 1) ----------------------

def test_CORD_20_거래량항은_전일대비_비율의_로그다():
    n = 21
    closes = 100.0 + 5.0 * np.sin(np.arange(n))
    vols = 1000.0 + 80.0 * np.cos(np.arange(n))
    df, i = synth_from_closes(closes, vols=vols)
    out = features_f3.compute_f3(df, i)

    close_e = closes
    vol_e = vols
    price_ret = (close_e[1:] - close_e[:-1]) / close_e[:-1]
    logv_ratio = np.log1p(vol_e[1:] / vol_e[:-1])  # log(V_t/V_{t-1} + 1)
    expected = np.corrcoef(price_ret, logv_ratio)[0, 1]
    assert out["CORD_20"] == pytest.approx(expected)

    # diff(log1p(V)) 를 쓰던 (틀린) 계산과는 값이 달라야 한다
    logv_wrong = np.diff(np.log1p(vol_e))
    wrong = np.corrcoef(price_ret, logv_wrong)[0, 1]
    assert out["CORD_20"] != pytest.approx(wrong)


# --- 중요5: SUMP_20 — 퍼센트 수익률이 아니라 원시 가격차 -------------------

def test_SUMP_20_원시_가격차를_쓴다():
    n = 21
    # 상승/하락 diff의 절댓값은 같은 패턴으로 반복하되(100, -90 반복),
    # 그 diff가 걸리는 종가 수준(분모)이 계속 달라지도록 누적시킨다 —
    # 그래야 퍼센트 버전과 원시차 버전의 gains/total 비율이 실제로 갈린다.
    diffs = np.tile([100.0, -90.0], (n - 1) // 2)
    closes = np.concatenate([[500.0], 500.0 + np.cumsum(diffs)])
    assert (closes > 0).all()
    df, i = synth_from_closes(closes)
    out = features_f3.compute_f3(df, i)

    raw_diff = np.diff(closes)
    expected = np.maximum(raw_diff, 0.0).sum() / (np.abs(raw_diff).sum() + 1e-12)
    assert out["SUMP_20"] == pytest.approx(expected)

    pct = raw_diff / closes[:-1]
    wrong = np.maximum(pct, 0.0).sum() / (np.abs(pct).sum() + 1e-12)
    assert out["SUMP_20"] != pytest.approx(wrong)


# --- 중요6: CNTP_20/SUMP_20/CORD_20/WVMA_20 은 d+1(21개) 원시 봉을 쓴다 -----

def test_CNTP_SUMP_는_d플러스1_윈도우로_계산한다():
    n = 25
    closes = np.full(n, 1000.0)
    closes[4] = 1.0  # d+1(21개) 창의 가장 오래된 봉만 극단적으로 낮게
    df, i = synth_from_closes(closes)  # i = n-1 = 24, lo=5, lo_e=4
    out = features_f3.compute_f3(df, i)
    # d+1(21개 원시봉)을 올바로 읽으면 index4→5 구간의 거대한 상승 diff가
    # 20개 diff 중 하나로 잡혀 CNTP=1/20, SUMP≈1.0이 되어야 한다. 이전
    # 버그(20개 원시봉만 읽어 19개 diff만 만듦)라면 그 diff가 통째로
    # 빠져 CNTP=0, SUMP=0이 된다.
    assert out["CNTP_20"] == pytest.approx(1 / 20)
    assert out["SUMP_20"] == pytest.approx(1.0)


# --- NaN(거래정지 봉)이 20봉 창 안에 섞인 경우 ------------------------------
# sources.mark_halted 규칙: 정지봉은 Open/High/Low/Volume=NaN, Close=직전값
# 유지. 이벤트 당일(i)은 events.is_breakout이 배제하므로 항상 유효하지만,
# 창의 다른 위치(i-19..i-1, CNTP/SUMP/CORD/WVMA는 i-20..i-1)에는 정지봉이
# 얼마든지 섞일 수 있다.

def test_RSV_20_창_안의_NaN은_최대최소_계산에서_제외된다():
    df, i = bar(100, 101, 99, 101)
    halt_i = i - 5
    df.loc[df.index[halt_i], ["Open", "High", "Low", "Volume"]] = np.nan
    out = features_f3.compute_f3(df, i)
    assert np.isfinite(out["RSV_20"])
    assert out["RSV_20"] == pytest.approx(1.0)


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
