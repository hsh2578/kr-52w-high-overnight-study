# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from surge_d1 import features_f4


def train_rows():
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
        "code": ["000001", "000002", "000003"],
        "ctype": ["정책·정부", "정책·정부", "테마·수혜"],
        "reason": ["정부 정책 수혜 기대", "정부 지원 정책 발표", "테마 관련주 부각"],
        "d1": [3.0, 1.0, -2.0],
    })


def test_F4_이름이_5개다():
    assert len(features_f4.F4_NAMES) == 5


def test_type_prior가_학습구간_평균이다():
    idx = features_f4.CatalystIndex()
    idx.fit(train_rows())
    out = idx.transform("정책·정부", "정부 정책 수혜")
    assert out["type_prior"] == pytest.approx(2.0)      # (3.0+1.0)/2


def test_모르는_유형은_전체평균으로_대체된다():
    idx = features_f4.CatalystIndex()
    idx.fit(train_rows())
    out = idx.transform("없는유형", "무언가")
    assert out["type_prior"] == pytest.approx((3.0 + 1.0 - 2.0) / 3)


def test_유사사례가_학습구간에서만_나온다():
    idx = features_f4.CatalystIndex()
    idx.fit(train_rows())
    out = idx.transform("정책·정부", "정부 정책 수혜 기대", k=2)
    assert out["analog_n"] == 2
    assert -5.0 <= out["analog_mean_d1"] <= 5.0
    assert 0.0 <= out["analog_win_rate"] <= 1.0


def test_재료가_없으면_중립값을_돌려준다():
    idx = features_f4.CatalystIndex()
    idx.fit(train_rows())
    out = idx.transform(None, None)
    assert out["analog_n"] == 0
    assert out["reason_len"] == 0.0


def test_fit_전에_transform하면_에러():
    idx = features_f4.CatalystIndex()
    with pytest.raises(RuntimeError):
        idx.transform("정책·정부", "정부 정책")


def test_attach_catalyst_날짜와_코드로_조인한다():
    ev = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
        "code": ["000001", "999999"],
    })
    cat = train_rows()
    out = features_f4.attach_catalyst(ev, cat)
    assert out.loc[0, "has_catalyst"] == 1
    assert out.loc[0, "ctype"] == "정책·정부"
    assert out.loc[1, "has_catalyst"] == 0


def test_fit이_본_행_밖의_데이터는_결과에_전혀_영향을_주지_않는다():
    """누수 방지 핵심 테스트.

    학습 구간(fit)에 없는, 미래(검정 구간)에만 존재하는 "진짜 유사한"(문자
    그대로 동일한) 사례를 질의(transform)해도 그 미래 사례의 d1 값이
    결과에 섞여 들면 안 된다. 즉 CatalystIndex는 fit()이 본 행만으로 계산된
    값을 돌려줘야 한다 — fit 이후에 더 가까운(심지어 완전히 일치하는)
    매치가 어딘가에 존재한다는 사실 자체를 몰라야 한다.

    train은 서로 다른 주제의 사례 4건으로 구성해(단순히 k가 전부를
    긁어오지 않도록) 코사인 유사도 랭킹이 실제로 의미를 가지게 한다.
    """
    train = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]),
        "code": ["000001", "000002", "000003", "000004"],
        "ctype": ["정책·정부", "테마·수혜", "실적·호재", "공시·계약"],
        "reason": ["정부 지원 정책 발표", "반도체 테마 관련주 부각",
                   "분기 실적 개선 발표", "대규모 수주 계약 체결"],
        "d1": [1.0, -2.0, 4.0, -1.0],
    })
    # 미래(학습 구간 밖) 사례: 질의문과 문자 그대로 동일한 reason/ctype이지만
    # 극단적인 d1(=100.0)을 가진다. fit에 이 행을 넘기지 않으므로 이 값이
    # analog_mean_d1/type_prior에 나타나면 누수다.
    future_exact_match = pd.DataFrame({
        "date": pd.to_datetime(["2025-06-01"]),
        "code": ["999999"],
        "ctype": ["정책·정부"],
        "reason": ["정부 정책 수혜 기대"],
        "d1": [100.0],
    })
    query_ctype, query_reason = "정책·정부", "정부 정책 수혜 기대"

    idx_without_future = features_f4.CatalystIndex()
    idx_without_future.fit(train)
    out_without_future = idx_without_future.transform(query_ctype, query_reason, k=1)

    idx_with_future = features_f4.CatalystIndex()
    idx_with_future.fit(pd.concat([train, future_exact_match], ignore_index=True))
    out_with_future = idx_with_future.transform(query_ctype, query_reason, k=1)

    # fit에 미래 행을 넣지 않았으면 극단값 100.0은 학습 구간 어디에도 없으므로
    # analog_mean_d1/type_prior 어느 쪽에도 나타날 수 없다.
    assert out_without_future["analog_mean_d1"] in {1.0, -2.0, 4.0, -1.0}
    assert out_without_future["analog_mean_d1"] != pytest.approx(100.0)
    assert out_without_future["type_prior"] == pytest.approx(1.0)

    # 반면 fit에 미래 행을 포함시키면(대조군) 문자 그대로 일치하는 사례라
    # 코사인 유사도 1.0으로 k=1 최근접이 그 미래 행이 되어 값이 새어나온다 —
    # 즉 이 테스트가 실제로 무언가를 검증하고 있음(민감함)을 증명한다.
    assert out_with_future["analog_mean_d1"] == pytest.approx(100.0)
    assert out_with_future["type_prior"] == pytest.approx(50.5)
