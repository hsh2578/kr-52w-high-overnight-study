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


# --- 추가: 재개봉(halt→resume) 오염 방지 (코드리뷰 지적, 수정) --------------
# mark_halted는 정지봉의 Open을 NaN으로 바꾸지만 Close는 직전값(정지 전
# 가격)을 그대로 들고 있다. 액면분할·감자 등으로 정지됐다가 재개되면
# 재개봉의 Open은 새 가격 기준의 진짜(유한한) 값이라, Open[t]/Close[t-1]에서
# t-1이 halted면 분모가 낡은 가격이 되어 가짜 초대형 갭이 isfinite를 통과해
# 버린다. halted 컬럼으로 그 항을 무조건 제외해야 한다.

def test_avg_gap_20d_재개봉_전일이_halted면_그_갭항을_제외한다():
    df = synth()
    df["halted"] = False
    i = 280
    halt_pos = i - 10          # 이 봉이 정지(따라서 Close가 낡은 값)
    resume_pos = halt_pos + 1  # 재개봉: t-1=halt_pos, t=resume_pos
    df.loc[df.index[halt_pos], "halted"] = True
    # 재개봉 Open을 정지 전 종가의 50배로 — 액면분할/감자 후 새 가격 기준의
    # 극단적이지만 '유한한' 값을 흉내낸다.
    df.loc[df.index[resume_pos], "Open"] = df["Close"].iloc[halt_pos] * 50.0

    out = features_f2.compute_f2(df, i, market_for(df))

    # 오염된 항(resume_pos)을 뺀 나머지 19개 원소로 기대값을 직접 계산한다.
    # (production 코드를 다시 호출하지 않고 독립적으로 재현 — 동어반복 방지)
    o = df["Open"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    idxs = [t for t in range(i - 19, i + 1) if t != resume_pos]
    expected_terms = np.array([o[t] / c[t - 1] - 1.0 for t in idxs])
    expected_mean = float(expected_terms.mean())
    expected_std = float(expected_terms.std())  # ddof=0

    contaminated = float(o[resume_pos] / c[halt_pos] - 1.0)
    assert contaminated == pytest.approx(49.0)  # 필터링 안 하면 이 값이 섞임

    assert out["avg_gap_20d"] == pytest.approx(expected_mean, rel=1e-9)
    assert out["gap_vol_20d"] == pytest.approx(expected_std, rel=1e-9)
    # 오염 항이 실제로 제외됐다는 방향성 증거: 포함됐다면 평균이 훨씬 커야 함
    assert out["avg_gap_20d"] < 1.0


def test_avg_gap_20d_창의_모든_전일이_halted면_0이_아니라_NaN을_반환한다():
    """분모(t-1)가 전부 halted면 유효 갭이 하나도 남지 않는다 — 이 경우도
    0.0이 아니라 NaN이어야 한다(Open 자체는 멀쩡히 유한해도 마찬가지)."""
    df = synth()
    i = 280
    df["halted"] = False
    # gaps 창의 분모는 c.iloc[i-20:i], 즉 위치 i-20..i-1 전부를 halted로.
    for k in range(i - 20, i):
        df.loc[df.index[k], "halted"] = True
    out = features_f2.compute_f2(df, i, market_for(df))
    assert np.isnan(out["avg_gap_20d"])
    assert np.isnan(out["gap_vol_20d"])


def test_avg_gap_20d_halted_컬럼이_전부_False면_컬럼없을때와_동일하다():
    """회귀 가드: halted 컬럼이 있어도 전부 False면(=깨끗한 창) halted
    컬럼이 아예 없을 때와 결과가 완전히 같아야 한다."""
    df_no_col = synth()
    df_all_false = synth()
    df_all_false["halted"] = False
    i = 280
    out_no_col = features_f2.compute_f2(df_no_col, i, market_for(df_no_col))
    out_all_false = features_f2.compute_f2(df_all_false, i, market_for(df_all_false))
    for k in features_f2.F2_NAMES:
        a, b = out_no_col[k], out_all_false[k]
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b)
        else:
            assert a == pytest.approx(b)


# --- 추가: gap_open/day_change의 NaN 폴백(prev_close<=0) 명시적 테스트 -----
# 이 브랜치는 브리프가 승인한 5개 피처(avg_gap_20d/gap_vol_20d/
# market_gap_prev/intraday_range/upper_shadow) 밖으로 확장한 것이다(코드
# 리뷰에서 지적). 철학적으로 일관되고(0.0은 "갭 0%"라는 잘못된 주장이 되므로)
# 실질적으로 죽은 분기이지만(정상 가격은 항상 양수), 명시적으로 테스트한다.

def test_gap_open_day_change_전일종가가_0이하이면_NaN을_반환한다():
    """prev_close<=0(정상 데이터에서는 사실상 발생하지 않는 방어적 케이스)
    이면 0.0으로 조용히 떨어뜨리지 않고 NaN을 반환해야 한다."""
    df = synth()
    i = 280
    df.loc[df.index[i - 1], "Close"] = 0.0
    out = features_f2.compute_f2(df, i, market_for(df))
    assert np.isnan(out["gap_open"])
    assert np.isnan(out["day_change"])


# --- 추가: prior_surge_5d 도 같은 재개봉 오염에 노출된다(자체 발견, 수정) ---
# t=i-5..i-1의 창은 HALT_LOOKBACK이 보장하는 무정지 구간(i-4..i+1) 밖이다.
# 실데이터 검증 중 086960(i=1499)에서 halted 재개봉 직후 일간수익률이
# +150%로 잡혀 급등 오탐(1.0)이 나는 것을 확인했다 — avg_gap_20d와 동일한
# 원인이므로 동일한 원칙(분모 t-1이 halted면 그 항 제외)을 적용한다.

def test_prior_surge_5d_전일이_halted면_그_항을_제외한다():
    df = synth()
    df["halted"] = False
    i = 280
    halt_pos = i - 3           # 창 안(t-1 = i-6..i-2)의 한 지점을 정지로
    resume_pos = halt_pos + 1  # t = halt_pos+1, 재개봉
    df.loc[df.index[halt_pos], "halted"] = True
    # 재개봉 종가를 정지 전 종가의 3배로 — 필터링 안 하면 300% "급등"으로
    # 오탐되어야 정상(테스트가 무의미해지지 않도록 충분히 크게).
    df.loc[df.index[resume_pos], "Close"] = df["Close"].iloc[halt_pos] * 3.0
    out = features_f2.compute_f2(df, i, market_for(df))
    assert out["prior_surge_5d"] == 0.0  # 오염 항이 제외되어 오탐이 없어야 함


def test_prior_surge_5d_필터링_안하면_오탐됐을_것임을_확인():
    """위 테스트의 전제 확인: halted 필터가 없다면 같은 데이터에서 실제로
    오탐(1.0)이 나왔을 것 — 그래야 위 테스트가 실제로 뭔가를 검증한 것."""
    df = synth()
    i = 280
    halt_pos = i - 3
    resume_pos = halt_pos + 1
    df.loc[df.index[resume_pos], "Close"] = df["Close"].iloc[halt_pos] * 3.0
    c = df["Close"].to_numpy(dtype=float)
    prior_unfiltered = c[i - 5:i] / c[i - 6:i - 1] - 1.0
    assert np.any(prior_unfiltered >= features_f2.SURGE_LEVEL)


def test_prior_surge_5d_halted_컬럼없어도_기존_동작_그대로():
    """회귀 가드: halted 컬럼이 아예 없는 df(기존 테스트들의 전제)에서는
    이전과 동일하게 동작해야 한다."""
    df = synth()
    i = 280
    df.loc[df.index[i - 3], "Close"] = df["Close"].iloc[i - 4] * 1.20
    out = features_f2.compute_f2(df, i, market_for(df))
    assert out["prior_surge_5d"] == 1.0
