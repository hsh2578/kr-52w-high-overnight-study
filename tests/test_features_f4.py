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


# ---------------------------------------------------------------------------
# 자기포함(self-inclusion) 누수 방지 — transform()의 exclude_key 파라미터
#
# fit()에 넘긴 행이 동시에 transform() 질의 대상이기도 한 경우(Task 11이
# 학습 폴드 자체의 피처를 만들 때 실제로 발생), exclude_key로 그 행을 자기
# 자신의 후보 풀에서 빼지 않으면 그 행의 라벨(d1)이 자기 자신의 피처로
# 그대로 새어 들어온다. 아래 테스트들은 이 문제를 재현하고 수정을 검증한다.
# ---------------------------------------------------------------------------

def self_incl_rows():
    return pd.DataFrame({
        "ctype": ["정책·정부", "정책·정부", "테마·수혜", "실적·호재", "공시·계약"],
        "reason": ["정부 정책 수혜 기대", "정부 지원 정책 발표", "반도체 테마 관련주 부각",
                   "분기 실적 개선 발표", "대규모 수주 계약 체결"],
        "d1": [999.0, 1.0, -2.0, 4.0, -1.0],
    }, index=[100, 101, 102, 103, 104])


def test_제외없이_질의하면_자기라벨이_analog에_그대로_남는다():
    """exclude_key를 안 주면 기존 동작(자기포함 누수)이 그대로 재현된다 —
    즉 이 수정이 옵트인이며, 차이가 실제로 관측 가능함을 보여준다."""
    idx = features_f4.CatalystIndex()
    idx.fit(self_incl_rows())
    out = idx.transform("정책·정부", "정부 정책 수혜 기대", k=1)
    assert out["analog_mean_d1"] == pytest.approx(999.0)
    assert out["type_prior"] == pytest.approx(500.0)   # (999+1)/2


def test_자기제외_켜면_자기라벨이_analog와_type_prior에서_빠진다():
    idx = features_f4.CatalystIndex()
    idx.fit(self_incl_rows())
    out = idx.transform("정책·정부", "정부 정책 수혜 기대", k=1, exclude_key=100)
    assert out["analog_mean_d1"] != pytest.approx(999.0)
    assert out["analog_mean_d1"] in {1.0, -2.0, 4.0, -1.0}
    assert out["analog_n"] == 1
    assert out["type_prior"] == pytest.approx(1.0)      # 999 제외, 남은 건 1.0 하나


def test_유일멤버_유형을_제외하면_전체평균으로_대체되고_NaN이_아니다():
    rows = pd.DataFrame({
        "ctype": ["단독유형", "테마·수혜", "실적·호재"],
        "reason": ["유일한 사유 문장", "반도체 테마 관련주 부각", "분기 실적 개선 발표"],
        "d1": [50.0, -2.0, 4.0],
    }, index=[200, 201, 202])
    idx = features_f4.CatalystIndex()
    idx.fit(rows)
    out = idx.transform("단독유형", "유일한 사유 문장", exclude_key=200)
    assert out["type_prior"] == pytest.approx(1.0)      # 전체(52) 중 자기(50) 뺀 나머지 평균 = (-2+4)/2
    assert not np.isnan(out["type_prior"])


def test_analog_n이_자기제외로_풀이_k와_같을때_1개_줄어든다():
    rows = pd.DataFrame({
        "ctype": ["정책·정부"] * 3,
        "reason": ["자기자신 문장", "비슷한 문장 하나", "비슷한 문장 둘"],
        "d1": [10.0, 1.0, 2.0],
    }, index=[300, 301, 302])
    idx = features_f4.CatalystIndex()
    idx.fit(rows)
    without = idx.transform("정책·정부", "자기자신 문장", k=3)
    with_excl = idx.transform("정책·정부", "자기자신 문장", k=3, exclude_key=300)
    assert without["analog_n"] == 3
    assert with_excl["analog_n"] == 2


def test_exclude_key가_fit집합에_없으면_기존동작과_동일하다():
    """검정 구간의 정상 경로: exclude_key로 자기 자신의 인덱스를 넘겨도
    그 값이 학습 구간(fit 집합)에 없으면 아무 것도 배제되지 않는다."""
    idx = features_f4.CatalystIndex()
    idx.fit(self_incl_rows())
    baseline = idx.transform("정책·정부", "정부 정책 수혜 기대", k=1)
    with_missing_key = idx.transform("정책·정부", "정부 정책 수혜 기대", k=1, exclude_key=999999)
    assert with_missing_key == baseline


# ---------------------------------------------------------------------------
# 중복 인덱스 라벨 — exclude_key 자기 배제를 무력화하는 조용한 실패 방지
#
# _key_to_pos가 라벨→위치 단일 매핑이라, fit()에 넘긴 rows.index에 중복
# 라벨이 있으면 같은 라벨의 두 번째 이후 행은 등록되지 않는다. 그 행을
# exclude_key로 질의해도 자기 자신이 제외되지 않고 자기 라벨이 그대로
# analog_mean_d1/type_prior로 새어 들어온다 — 에러 없이 조용히. fit()이
# 이런 인덱스를 아예 거부하도록(시끄럽게 실패하도록) 한다.
# ---------------------------------------------------------------------------

def test_중복된_인덱스로_fit하면_에러가_나고_라벨을_알려준다():
    rows = pd.DataFrame({
        "ctype": ["정책·정부", "정책·정부", "실적·호재"],
        "reason": ["첫번째 사유 문장", "두번째 사유 문장", "세번째 사유 문장"],
        "d1": [10.0, 20.0, 30.0],
    }, index=[5, 5, 6])
    idx = features_f4.CatalystIndex()
    with pytest.raises(ValueError, match="5"):
        idx.fit(rows)


def test_유일한_인덱스면_fit이_에러없이_기존과_동일하게_동작한다():
    idx = features_f4.CatalystIndex()
    idx.fit(self_incl_rows())  # index=[100, 101, 102, 103, 104], 모두 유일
    out = idx.transform("정책·정부", "정부 정책 수혜 기대", k=1, exclude_key=100)
    assert out["analog_mean_d1"] != pytest.approx(999.0)
    assert out["type_prior"] == pytest.approx(1.0)
