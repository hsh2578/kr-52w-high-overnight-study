# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from surge_d1 import (dataset, features_f1, features_f2, features_f3,
                      features_f4, features_f5)


def test_FEATURE_SETS가_누적구조다():
    fs = dataset.FEATURE_SETS
    assert fs["M1"] == features_f1.F1_NAMES
    assert fs["M2"] == features_f1.F1_NAMES + features_f2.F2_NAMES
    assert fs["M3"] == (features_f1.F1_NAMES + features_f2.F2_NAMES
                        + features_f3.F3_NAMES)
    assert fs["M4"] == fs["M3"] + features_f5.F5_NAMES     # +수급
    assert fs["M5"] == fs["M4"] + features_f4.F4_NAMES     # +재료


def test_피처개수가_스펙과_일치한다():
    fs = dataset.FEATURE_SETS
    assert len(fs["M1"]) == 14
    assert len(fs["M2"]) == 24
    assert len(fs["M3"]) == 44
    assert len(fs["M4"]) == 50
    assert len(fs["M5"]) == 55


def test_피처이름에_중복이_없다():
    names = dataset.FEATURE_SETS["M5"]
    assert len(names) == len(set(names))


def test_라벨은_절대_피처에_없다():
    """d1(재료 CSV의 익일 수익률)과 우리 라벨이 피처로 새면 결과가 전부 무효다."""
    banned = {"d1", "y_raw", "y_hold", "y_hold20", "y_mkt", "y_cls", "y_up",
              "close", "i"}
    for name, feats in dataset.FEATURE_SETS.items():
        assert not (set(feats) & banned), (name, set(feats) & banned)


def test_add_sample_weight_그날_이벤트수의_역수():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 4 + ["2024-01-03"]),
    })
    out = dataset.add_sample_weight(df)
    assert list(out["weight"].round(6)) == [0.25, 0.25, 0.25, 0.25, 1.0]


def test_build_dataset이_라벨누수를_감지하면_예외를_던진다():
    """BANNED_AS_FEATURE에 든 이름이 피처셋에 섞이면 build_dataset이 즉시 실패해야 한다.

    이 테스트는 그 가드가 실제로 작동함을 증명한다: 무해한 피처셋에
    금지된 컬럼명을 임시로 주입하고 build_dataset이 RuntimeError를
    던지는지 확인한다.
    """
    original = dataset.FEATURE_SETS["M1"]
    try:
        dataset.FEATURE_SETS["M1"] = list(original) + ["y_raw"]
        with pytest.raises(RuntimeError):
            dataset.build_dataset()
    finally:
        dataset.FEATURE_SETS["M1"] = original
