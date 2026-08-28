# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from surge_d1 import report


def toy_events(n=300):
    rng = np.random.default_rng(0)
    dates = pd.to_datetime("2023-01-02") + pd.to_timedelta(
        rng.integers(0, 900, n), unit="D")
    return pd.DataFrame({
        "date": dates,
        # write_report 가 종목 수와 재료 매칭률을 보고하므로 픽스처에도 있어야 한다
        "code": [f"{i % 37:06d}" for i in range(n)],
        "has_catalyst": rng.integers(0, 2, n),
        "y_raw": rng.normal(0.003, 0.03, n),
        "y_hold": rng.normal(0.001, 0.06, n),
        "y_hold20": rng.normal(0.012, 0.28, n),
        "y_cls": rng.integers(0, 2, n),
        "y_up": rng.integers(0, 2, n),
    })


def toy_backtest_stats():
    return pd.DataFrame({
        "strategy": ["model_top5", "model_top10", "all_breakouts", "random_5"],
        "cum_return": [0.42, 0.31, 0.18, 0.17],
        "cagr": [0.15, 0.11, 0.07, 0.066],
        "vol": [0.22, 0.19, 0.14, 0.21],
        "sharpe": [0.68, 0.58, 0.50, 0.31],
        "mdd": [0.18, 0.15, 0.12, 0.19],
        "win_rate": [0.47, 0.46, 0.44, 0.44],
        "n_days": [700] * 4,
        "n_trades": [3500, 7000, 21000, 3500],
    })


def toy_ablation():
    return pd.DataFrame({
        "model": ["M0", "M1", "M2", "M3"],
        "fold": [1, 1, 1, 1],
        "test_year": [2024] * 4,
        "sample": ["full"] * 4,
        "auc": [0.50, 0.52, 0.55, 0.57],
        "brier": [0.25, 0.24, 0.24, 0.23],
        "ic": [0.0, 0.01, 0.02, 0.03],
        "ic_t": [0.0, 1.0, 2.0, 3.0],
        "top_decile": [0.003, 0.004, 0.006, 0.007],
        "n": [100] * 4,
    })


def test_fig_yearly가_파일을_만든다(tmp_path):
    p = tmp_path / "yearly.png"
    report.fig_yearly(toy_events(), p)
    assert p.exists() and p.stat().st_size > 0


def test_fig_return_distribution이_파일을_만든다(tmp_path):
    p = tmp_path / "dist.png"
    report.fig_return_distribution(toy_events(), p)
    assert p.exists() and p.stat().st_size > 0


def test_fig_ablation이_파일을_만든다(tmp_path):
    p = tmp_path / "ab.png"
    report.fig_ablation(toy_ablation(), p)
    assert p.exists() and p.stat().st_size > 0


def test_write_report가_핵심수치를_포함한다(tmp_path):
    p = tmp_path / "r.md"
    ev = toy_events()
    lk = pd.DataFrame({"method": ["shuffle_kfold", "walk_forward"],
                       "auc": [0.60, 0.55], "ic": [0.05, 0.02],
                       "auc_bias": [0.05, 0.0]})
    sens = pd.DataFrame({"extra_cost": [0.0, 0.002],
                         "cum_return": [0.42, 0.05], "cagr": [0.15, 0.02],
                         "sharpe": [0.68, 0.11], "mdd": [0.18, 0.24],
                         "win_rate": [0.47, 0.43]})
    report.write_report(ev, toy_ablation(), lk, toy_backtest_stats(), sens, p)
    text = p.read_text(encoding="utf-8")
    assert "이벤트" in text
    assert "M3" in text
    assert str(len(ev)) in text
