# -*- coding: utf-8 -*-
"""F2(D+1 특화 피처 10개) 테스트. Task 5 브리프 §Step1 + 정렬/NaN 보강."""
import numpy as np
import pandas as pd
import pytest

from surge_d1 import features_f2


def synth(n=300):
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = np.linspace(100, 200, n)
    return pd.DataFrame(
        {"Open": c * 0.99, "High": c * 1.02, "Low": c * 0.98,
         "Close": c, "Volume": np.full(n, 1000.0)},
        index=idx,
    )


def market_for(df):
    return pd.DataFrame({"ret": 0.0, "gap": 0.01, "n": 1}, index=df.index)


def test_F2_이름이_10개고_반환키와_일치한다():
    df = synth()
    out = features_f2.compute_f2(df, 280, market_for(df))
    assert len(features_f2.F2_NAMES) == 10
    assert set(out.keys()) == set(features_f2.F2_NAMES)


def test_gap_open_당일시가갭():
    df = synth()
    i = 280
    df.loc[df.index[i - 1], "Close"] = 100.0
    df.loc[df.index[i], "Open"] = 103.0
    out = features_f2.compute_f2(df, i, market_for(df))
    assert out["gap_open"] == pytest.approx(0.03)


def test_day_change_당일등락률():
    df = synth()
    i = 280
    df.loc[df.index[i - 1], "Close"] = 100.0
    df.loc[df.index[i], "Close"] = 112.0
    out = features_f2.compute_f2(df, i, market_for(df))
    assert out["day_change"] == pytest.approx(0.12)


def test_upper_shadow_윗꼬리비율():
    df = synth()
    i = 280
    df.loc[df.index[i], "High"] = 200.0
    df.loc[df.index[i], "Low"] = 100.0
    df.loc[df.index[i], "Close"] = 160.0
    out = features_f2.compute_f2(df, i, market_for(df))
    assert out["upper_shadow"] == pytest.approx(0.4)   # (200-160)/(200-100)


def test_aligned_정배열이면_1():
    df = synth()   # 단조 증가라 C > MA20 > MA60
    out = features_f2.compute_f2(df, 280, market_for(df))
    assert out["aligned"] == 1.0


def test_prior_surge_5d_직전5봉에_급등이_있으면_1():
    df = synth()
    i = 280
    df.loc[df.index[i - 3], "Close"] = df["Close"].iloc[i - 4] * 1.20
    out = features_f2.compute_f2(df, i, market_for(df))
    assert out["prior_surge_5d"] == 1.0


def test_dow_요일이_0에서_4():
    df = synth()
    out = features_f2.compute_f2(df, 280, market_for(df))
    assert 0 <= out["dow"] <= 4


def test_market_gap_prev_직전일_시장갭():
    df = synth()
    i = 280
    mk = market_for(df)
    mk.loc[df.index[i - 1], "gap"] = 0.007
    out = features_f2.compute_f2(df, i, mk)
    assert out["market_gap_prev"] == pytest.approx(0.007)


# --- 추가: 정렬(off-by-one) 검증 -------------------------------------------
# 한 칸 밀린 정렬로도 통과하는 테스트는 아무것도 증명하지 못한다(브리프 지시).
# avg_gap_20d/gap_vol_20d 창은 Open[t]/Close[t-1]을 t=i-19..i(20개, 당일 포함)
# 에 대해 구한다. i-3에만 큰 갭을 심어 정확히 그 위치가 반영되는지 확인한다.

def test_avg_gap_20d_당일포함_20봉창_정렬():
    df = synth()
    i = 280
    # 창(i-19..i) 안의 한 지점(i-3)에만 10% 갭을 심는다.
    df.loc[df.index[i - 3], "Open"] = df["Close"].iloc[i - 4] * 1.10
    out = features_f2.compute_f2(df, i, market_for(df))

    gaps = (df["Open"].iloc[i - 19:i + 1].to_numpy() /
            df["Close"].iloc[i - 20:i].to_numpy()) - 1.0
    expected = float(np.mean(gaps))
    assert out["avg_gap_20d"] == pytest.approx(expected, rel=1e-9)


def test_avg_gap_20d_당일_gap_open이_창의_마지막_항이다():
    """avg_gap_20d 창의 마지막 원소는 당일(i)의 gap_open과 같은 값이어야
    한다 — Open[t]/Close[t-1] 정렬이 t=i에서 어긋나지 않았는지 확인."""
    df = synth()
    i = 280
    df.loc[df.index[i - 1], "Close"] = 150.0
    df.loc[df.index[i], "Open"] = 165.0  # gap_open = 0.10
    out = features_f2.compute_f2(df, i, market_for(df))
    assert out["gap_open"] == pytest.approx(0.10)

    # 창의 다른 19개 항을 전부 0으로 만들면(Open == 전일 Close) 평균은
    # gap_open/20이 되어야 한다 — 마지막 항이 정확히 당일 gap_open임을 증명.
    for k in range(i - 19, i):
        df.loc[df.index[k], "Open"] = df["Close"].iloc[k - 1]
    out2 = features_f2.compute_f2(df, i, market_for(df))
    assert out2["avg_gap_20d"] == pytest.approx(0.10 / 20, rel=1e-9)


def test_prior_surge_5d_한칸_밀리면_감지못한다():
    """prior_surge_5d 창은 t=i-5..i-1(당일 제외 직전 5봉)이어야 한다.
    i-6에 급등을 심으면(창 밖) 감지되지 않아야 하고, i-1에 심으면(창 안,
    경계) 감지되어야 한다 — 이 둘의 결과가 달라야 정렬이 검증된다."""
    df_out = synth()
    df_in = synth()
    i = 280
    df_out.loc[df_out.index[i - 6], "Close"] = df_out["Close"].iloc[i - 7] * 1.30
    df_in.loc[df_in.index[i - 1], "Close"] = df_in["Close"].iloc[i - 2] * 1.30
    out_out = features_f2.compute_f2(df_out, i, market_for(df_out))
    out_in = features_f2.compute_f2(df_in, i, market_for(df_in))
    assert out_out["prior_surge_5d"] == 0.0
    assert out_in["prior_surge_5d"] == 1.0


def test_prior_surge_5d_당일급등은_보지않는다():
    """당일(i) 등락은 day_change로 이미 잡히므로 prior_surge_5d 창에는
    포함되지 않아야 한다(당일 제외, 직전 5봉만)."""
    df = synth()
    i = 280
    df.loc[df.index[i], "Close"] = df["Close"].iloc[i - 1] * 1.30
    out = features_f2.compute_f2(df, i, market_for(df))
    assert out["prior_surge_5d"] == 0.0


# --- 추가: NaN(거래정지) 처리 -----------------------------------------------

def test_avg_gap_20d_창안에_정지봉_NaN이있으면_건너뛴다():
    """20봉 창 안의 일부 Open이 NaN이어도(거래정지) 에러 없이 나머지로
    평균을 낸다 — 0.0으로 조용히 대체되면 안 된다(값이 달라져야 함)."""
    df = synth()
    i = 280
    baseline = features_f2.compute_f2(df, i, market_for(df))["avg_gap_20d"]
    df.loc[df.index[i - 10], "Open"] = np.nan
    out = features_f2.compute_f2(df, i, market_for(df))
    assert np.isfinite(out["avg_gap_20d"])
    # NaN 제외 평균이므로 정확히 baseline과 같지는 않되(원소 수가 줄어듦),
    # 여전히 유한하고 baseline과 아주 가까운 값이어야 한다(합성데이터가
    # 거의 동일한 갭을 갖기 때문).
    assert out["avg_gap_20d"] == pytest.approx(baseline, abs=1e-3)


def test_avg_gap_20d_창전체가_NaN이면_0이_아니라_NaN을_반환한다():
    """창의 20개 원소가 전부 결측이면 '평균 갭 0%'라는 잘못된 주장을 하는
    대신 명시적으로 NaN을 반환해야 한다."""
    df = synth()
    i = 280
    for k in range(i - 19, i + 1):
        df.loc[df.index[k], "Open"] = np.nan
    out = features_f2.compute_f2(df, i, market_for(df))
    assert np.isnan(out["avg_gap_20d"])
    assert np.isnan(out["gap_vol_20d"])


def test_gap_vol_20d_유효원소가_하나뿐이면_0을_반환한다():
    """창에 유효한 갭이 딱 1개면(표본분산 미정의) 0.0을 반환한다 —
    결측이 아니라 '변동 없음'으로 간주 가능한 경계 케이스."""
    df = synth()
    i = 280
    for k in range(i - 19, i + 1):
        df.loc[df.index[k], "Open"] = np.nan
    df.loc[df.index[i], "Open"] = df["Close"].iloc[i - 1] * 1.05
    out = features_f2.compute_f2(df, i, market_for(df))
    assert out["gap_vol_20d"] == pytest.approx(0.0)
    assert out["avg_gap_20d"] == pytest.approx(0.05)


def test_intraday_range_당일_High가_NaN이면_NaN을_반환한다():
    """이벤트 당일(i)은 정지가 배제되지만, 방어적으로 High/Low가 NaN이면
    intraday_range/upper_shadow가 조용히 0.0이 되지 않고 NaN이어야 한다."""
    df = synth()
    i = 280
    df.loc[df.index[i], "High"] = np.nan
    out = features_f2.compute_f2(df, i, market_for(df))
    assert np.isnan(out["intraday_range"])
    assert np.isnan(out["upper_shadow"])


def test_upper_shadow_고가저가같으면_0을_반환한다():
    """High == Low(레인지 0, 결측 아님)면 그림자는 정의상 0.0."""
    df = synth()
    i = 280
    df.loc[df.index[i], "High"] = 150.0
    df.loc[df.index[i], "Low"] = 150.0
    df.loc[df.index[i], "Close"] = 150.0
    out = features_f2.compute_f2(df, i, market_for(df))
    assert out["upper_shadow"] == pytest.approx(0.0)
    assert out["intraday_range"] == pytest.approx(0.0)


def test_market_gap_prev_직전일_시장갭이_NaN이면_NaN을_반환한다():
    """시장 지수 자체가 그날 계산 불가(n==0)로 NaN인 경우, 0.0으로
    조용히 대체하면 '그날 시장 갭 0%'라는 잘못된 주장이 된다."""
    df = synth()
    i = 280
    mk = market_for(df)
    mk.loc[df.index[i - 1], "gap"] = np.nan
    out = features_f2.compute_f2(df, i, mk)
    assert np.isnan(out["market_gap_prev"])
