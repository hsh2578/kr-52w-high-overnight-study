# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from surge_d1 import models


def toy(n=400, seed=0):
    """f1이 y와 상관 있는 장난감 데이터."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    y = 0.02 * f1 + rng.normal(scale=0.01, size=n)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 200, n), unit="D")
    return pd.DataFrame({
        "f1": f1, "f2": f2, "y_raw": y, "y_cls": (y > 0).astype(int),
        "date": dates, "weight": 1.0,
    })


def test_fit_predict가_세가지_예측을_돌려준다():
    df = toy()
    tr, te = df.iloc[:300], df.iloc[300:]
    out = models.fit_predict(tr, te, ["f1", "f2"])
    assert set(out.keys()) == {"pred_reg", "prob_raw", "prob_cal"}
    for k in out:
        assert len(out[k]) == len(te)


def test_보정확률이_0과_1_사이다():
    df = toy()
    out = models.fit_predict(df.iloc[:300], df.iloc[300:], ["f1", "f2"])
    assert out["prob_cal"].min() >= 0.0
    assert out["prob_cal"].max() <= 1.0


def test_evaluate가_필요한_지표를_모두_낸다():
    df = toy()
    te = df.iloc[300:].reset_index(drop=True)
    out = models.fit_predict(df.iloc[:300], te, ["f1", "f2"])
    m = models.evaluate(te, out)
    for k in ("auc", "brier", "ic", "ic_t", "top_decile", "n"):
        assert k in m
    assert 0.0 <= m["auc"] <= 1.0
    assert 0.0 <= m["brier"] <= 1.0


def test_유효한_피처가_있으면_AUC가_0_5를_넘는다():
    df = toy(n=1200, seed=1)
    te = df.iloc[900:].reset_index(drop=True)
    out = models.fit_predict(df.iloc[:900], te, ["f1", "f2"])
    assert models.evaluate(te, out)["auc"] > 0.5


def test_M0는_상수예측이다():
    df = toy()
    te = df.iloc[300:].reset_index(drop=True)
    out = models.fit_predict(df.iloc[:300], te, [])
    assert np.unique(out["pred_reg"]).size == 1


def test_build_f4가_학습행에_exclude_key를_넘겨_자기누수를_막는다():
    """CatalystIndex.fit()한 학습 폴드 자신의 행에 심어둔 d1=999가
    그 행 자신의 F4 피처(analog_mean_d1/type_prior)로 새어 들어오지
    않아야 한다 — exclude_key를 빠뜨리면 조용히 새어 들어온다."""
    n = 20
    tr = pd.DataFrame({
        "ctype": ["공시"] * n,
        "reason": [f"동일한 재료 사유 문구 {i % 3}" for i in range(n)],
        "d1": [999.0] + [0.01 * i for i in range(1, n)],
        "has_catalyst": 1,
    })
    te = pd.DataFrame({
        "ctype": ["공시"],
        "reason": ["동일한 재료 사유 문구 0"],
        "has_catalyst": [1],
    })
    tr4, te4 = models._build_f4(tr, te)
    # 자기 자신(첫 행, d1=999)의 analog_mean_d1/type_prior에는 999가
    # 섞이면 안 된다 — 나머지 값들의 범위(0~0.19) 안에 있어야 정상.
    self_row = tr4.iloc[0]
    assert self_row["analog_mean_d1"] < 5.0
    assert self_row["type_prior"] < 5.0


def test_build_f4가_원본_인덱스를_유지한_프레임에서도_동작한다():
    """reset_index 하지 않은(구멍난) 인덱스에서도 exclude_key가 제대로
    매칭돼야 한다."""
    tr = pd.DataFrame({
        "ctype": ["공시", "실적"],
        "reason": ["재료 문구 A", "재료 문구 B"],
        "d1": [999.0, 0.02],
        "has_catalyst": [1, 1],
    }, index=[7, 15])
    te = pd.DataFrame({
        "ctype": ["공시"],
        "reason": ["재료 문구 A"],
        "has_catalyst": [1],
    }, index=[42])
    tr4, te4 = models._build_f4(tr, te)
    assert tr4.iloc[0]["analog_mean_d1"] < 5.0


# --- 리뷰 반영: assert_train_before_test (Minor 4) ---

def test_assert_train_before_test가_정상_분할은_통과시킨다():
    tr = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-05"])})
    te = pd.DataFrame({"date": pd.to_datetime(["2024-02-01", "2024-02-02"])})
    models.assert_train_before_test(tr, te)  # 예외 없이 통과해야 한다


def test_assert_train_before_test가_시간_역전을_잡아낸다():
    tr = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-03-01"])})
    te = pd.DataFrame({"date": pd.to_datetime(["2024-02-01", "2024-02-15"])})
    with pytest.raises(ValueError):
        models.assert_train_before_test(tr, te)


def test_assert_train_before_test가_빈_프레임은_통과시킨다():
    tr = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})
    te = pd.DataFrame({"date": pd.to_datetime([])})
    models.assert_train_before_test(tr, te)
    models.assert_train_before_test(te, tr)


# --- 리뷰 반영: evaluate()가 상수 예측에 IC를 NaN으로 낸다 (Important 1) ---

def test_evaluate가_예측이_상수면_ic를_NaN으로_낸다():
    df = toy()
    te = df.iloc[300:].reset_index(drop=True)
    out = {"pred_reg": np.zeros(len(te)), "prob_raw": np.full(len(te), 0.5),
          "prob_cal": np.full(len(te), 0.5)}
    m = models.evaluate(te, out)
    assert np.isnan(m["ic"])
    assert np.isnan(m["ic_t"])


def test_evaluate가_예측에_변동이_있으면_ic가_숫자다():
    df = toy(n=1200, seed=1)
    te = df.iloc[900:].reset_index(drop=True)
    out = models.fit_predict(df.iloc[:900], te, ["f1", "f2"])
    m = models.evaluate(te, out)
    assert not np.isnan(m["ic"])


# --- 리뷰 반영: 학습/검정 예측을 함께 내는 내부 함수 (과적합 진단용) ---

def test_fit_predict_train_test가_학습_검정_예측을_모두_낸다():
    df = toy(n=1200, seed=1)
    tr = df.iloc[:900].reset_index(drop=True)
    te = df.iloc[900:].reset_index(drop=True)
    train_out, test_out = models._fit_predict_train_test(tr, te, ["f1", "f2"])
    assert len(train_out["pred_reg"]) == len(tr)
    assert len(test_out["pred_reg"]) == len(te)
    for out in (train_out, test_out):
        assert set(out.keys()) == {"pred_reg", "prob_raw", "prob_cal"}


def test_fit_predict가_fit_predict_train_test의_검정쪽과_일치한다():
    df = toy(n=1200, seed=1)
    tr = df.iloc[:900].reset_index(drop=True)
    te = df.iloc[900:].reset_index(drop=True)
    out1 = models.fit_predict(tr, te, ["f1", "f2"])
    _, out2 = models._fit_predict_train_test(tr, te, ["f1", "f2"])
    assert np.allclose(out1["pred_reg"], out2["pred_reg"])


# --- 리뷰 반영: 부트스트랩 CI·p값 헬퍼 (Important 2) ---

def test_bootstrap_delta_stats가_평균과_CI와_p값을_낸다():
    vals = [0.01, 0.02, 0.015, 0.005]
    out = models._bootstrap_delta_stats(vals, n=200)
    assert out["delta"] == pytest.approx(np.mean(vals))
    assert out["ci_lo"] <= out["delta"] <= out["ci_hi"]
    assert 0.0 <= out["p_value"] <= 1.0


def test_bootstrap_paired_auc_delta가_필요한_키를_모두_낸다():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    a = rng.uniform(size=200)
    b = np.clip(a + 0.2 * (y - 0.5), 0, 1)  # b가 y와 더 잘 맞도록 구성
    out = models.bootstrap_paired_auc_delta(y, a, b, n=200)
    for k in ("delta", "ci_lo", "ci_hi", "p_value", "frac_a_better", "n_boot"):
        assert k in out
    assert out["ci_lo"] <= out["delta"] <= out["ci_hi"]
    assert out["delta"] > 0  # b가 뚜렷이 더 나으므로 양의 델타여야 한다
