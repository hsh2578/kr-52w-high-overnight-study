# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from surge_d1 import tuning


def toy(n_days=6, per_day=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        # 날짜마다 스케일이 다르게 만들어 횡단면 정규화의 효과가 드러나게 한다
        base = 100.0 * (d + 1)
        for j in range(per_day):
            rows.append({"date": date, "f1": base + rng.normal(),
                         "f2": rng.normal(), "y_raw": rng.normal() * 0.01,
                         "weight": 1.0 / per_day})
    df = pd.DataFrame(rows)
    df["y_cls"] = (df["y_raw"] > 0).astype(int)
    return df


def test_횡단면정규화는_날짜마다_평균0_분산1로_만든다():
    df = toy()
    out = tuning.cross_sectional_normalize(df, ["f1", "f2"], method="zscore")
    for _, g in out.groupby("date"):
        assert abs(g["f1"].mean()) < 1e-9
        assert abs(g["f1"].std(ddof=0) - 1.0) < 1e-6


def test_횡단면정규화가_날짜간_스케일차이를_제거한다():
    df = toy()
    # 원본은 날짜별 평균이 100,200,...으로 크게 다르다
    raw_spread = df.groupby("date")["f1"].mean().std()
    out = tuning.cross_sectional_normalize(df, ["f1"], method="zscore")
    norm_spread = out.groupby("date")["f1"].mean().std()
    assert raw_spread > 50
    assert norm_spread < 1e-9


def test_rank정규화는_0과1_사이다():
    df = toy()
    out = tuning.cross_sectional_normalize(df, ["f1"], method="rank")
    assert out["f1"].min() >= 0.0
    assert out["f1"].max() <= 1.0
    for _, g in out.groupby("date"):
        assert g["f1"].nunique() == len(g)      # 순위이므로 동점이 없어야 한다


def test_정규화는_라벨을_건드리지_않는다():
    df = toy()
    out = tuning.cross_sectional_normalize(df, ["f1", "f2"], method="zscore")
    pd.testing.assert_series_equal(out["y_raw"], df["y_raw"])
    pd.testing.assert_series_equal(out["y_cls"], df["y_cls"])


def test_inner_tune은_검정데이터를_보지_않는다(monkeypatch):
    """전달된 학습 프레임 밖의 행이 결과에 영향을 주면 안 된다."""
    df = toy(n_days=12, per_day=12)
    train = df[df["date"] < pd.Timestamp("2024-01-09")]
    grid = {"max_depth": [2, 3], "learning_rate": [0.05]}
    best_a = tuning.inner_tune(train, ["f1", "f2"], grid, n_splits=2)
    # 학습 프레임은 그대로 두고 그 뒤 구간만 값을 오염시킨다
    poisoned = df.copy()
    mask = poisoned["date"] >= pd.Timestamp("2024-01-09")
    poisoned.loc[mask, ["f1", "f2", "y_raw", "y_cls"]] = 999.0
    train_p = poisoned[poisoned["date"] < pd.Timestamp("2024-01-09")]
    best_b = tuning.inner_tune(train_p, ["f1", "f2"], grid, n_splits=2)
    assert best_a == best_b


def test_inner_tune의_내부분할도_시간순이다():
    """내부 검증 분할이 무작위면 시계열 튜닝의 의미가 없다."""
    df = toy(n_days=12, per_day=8)
    splits = tuning.inner_time_splits(df["date"], n_splits=3)
    assert len(splits) == 3
    d = pd.to_datetime(df["date"])
    for tr, va in splits:
        assert d[tr].max() < d[va].min()


def test_inner_tune은_그리드안의_값을_돌려준다():
    df = toy(n_days=12, per_day=12)
    grid = {"max_depth": [2, 4], "learning_rate": [0.05, 0.1]}
    best = tuning.inner_tune(df, ["f1", "f2"], grid, n_splits=2)
    assert best["max_depth"] in grid["max_depth"]
    assert best["learning_rate"] in grid["learning_rate"]
