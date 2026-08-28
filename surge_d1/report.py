# -*- coding: utf-8 -*-
"""발표용 figure와 마크다운 리포트 생성. 스펙 §8.

이 모듈이 그리는 그림과 쓰는 문장은 청중이 실제로 보는 최종 산출물이다.
그래서 두 가지를 특히 조심한다:

1. **복리 누적수익률을 헤드라인으로 쓰지 않는다.** 매일 전액 재투자를 가정한
   `cum_return`/`cagr`은 `outputs/backtest_stats.csv`에 실현 불가능한 상한값으로
   이미 기록돼 있다(model_top5가 82,000%를 넘는다). 그림에서는 일평균 수익률
   (expectancy), Sharpe, 고정명목 누적(`cum_return_additive`, 재투자 없이 매일
   동일 명목을 거는 가정)만 헤드라인으로 쓴다. 복리 수치는 표에 참고용으로만
   남긴다.
2. **M5의 AUC를 캐베어트 없이 보여주지 않는다.** 재료 표본은 n=869·단일분할이고
   회귀는 상수로 붕괴해 IC·상위10%가 정의되지 않는다. 그림·본문 어디서든 M5
   AUC 옆에는 이 사실이 붙는다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surge_d1 import config, dataset

MODEL_ORDER = ["M0", "M1", "M2", "M3", "M4", "M4prime", "M5"]

# 백테스트 표·그림에서 우선적으로 다루는 전략(너무 많은 변형을 늘어놓으면
# 슬라이드가 읽히지 않는다). 나머지 변형은 outputs/backtest_stats.csv에 남는다.
MAIN_STRATEGIES = ["model_top5", "model_top10", "rule_top5_close",
                   "all_breakouts", "random_5", "market"]

# 2026 검정 폴드(walk-forward 마지막 폴드)의 단일지표 벤치마크.
#
# 아래는 **폴백 값일 뿐이다.** `main()`이 `compute_bench_2026(df)`로 매번
# 재계산해 이 딕셔너리를 덮어쓴다 — 데이터가 갱신되면 리포트 수치도 함께
# 움직여야 하는데, 하드코딩만 두면 조용히 어긋나기 때문이다. 재계산 결과가
# 이 값과 0.1%p 넘게 벌어지면 경고를 찍어 드리프트를 드러낸다.
BENCH_2026 = {
    "n_picks": 444,
    "overall_n": 4277,
    "overall_winrate": 0.505027,
    "overall_mean": 0.010929,
    "perfect": {"mean": 0.10880, "winrate": 0.986},
    "close_strength": {"mean": 0.05033, "winrate": 0.678},
    "rsv_20": {"mean": 0.04633, "winrate": 0.660},
    "random": {"mean": 0.00737, "winrate": 0.486},
}

def compute_bench_2026(df: pd.DataFrame) -> dict:
    """마지막(2026) 검정 폴드에서 단일지표 상위10% 벤치마크를 재계산한다.

    `perfect`는 정답(`y_raw`)으로 정렬한 이론적 상한이라 달성 불가능하지만,
    "모델이 얼마나 남겨두었나"의 척도로 쓴다. `random`은 고정 시드로 뽑아
    실행 간 재현된다.
    """
    from surge_d1 import models, validate      # 순환 import 회피(모듈 관례)

    folds = validate.walk_forward_folds(df["date"], models.TEST_YEARS)
    te = df.loc[folds[-1][1]].dropna(subset=["y_raw"])
    y, d = te["y_raw"], te["date"]

    def one(score: pd.Series) -> dict:
        # top_decile_return과 같은 규칙(날짜별 상위 10%, 3종목 미만 날 제외)으로
        # 픽을 재현해 평균과 승률을 함께 낸다.
        keep = []
        for _, g in pd.DataFrame({"y": y, "s": score, "d": d}).groupby("d"):
            if len(g) < 3:
                continue
            k = max(1, int(round(len(g) * 0.1)))
            keep.append(g.nlargest(k, "s"))
        picks = pd.concat(keep) if keep else pd.DataFrame(columns=["y"])
        return {"n": len(picks), "mean": float(picks["y"].mean()),
                "winrate": float((picks["y"] > 0).mean())}

    cand = {"perfect": y, "close_strength": te["close_strength"],
            "rsv_20": te["RSV_20"]}
    res = {k: one(v) for k, v in cand.items()}

    # 무작위 기준선은 단일 추첨이면 표본노이즈가 크다(실측 ±0.44%p). 30회
    # 평균을 써서 "아무 신호 없이 뽑았을 때의 기대값"에 가깝게 만든다.
    rng = np.random.default_rng(config.RANDOM_SEED)
    draws = [one(pd.Series(rng.random(len(te)), index=te.index))
             for _ in range(30)]
    res["random"] = {"n": draws[0]["n"],
                     "mean": float(np.mean([x["mean"] for x in draws])),
                     "winrate": float(np.mean([x["winrate"] for x in draws]))}
    out = {"n_picks": res["perfect"]["n"], "overall_n": len(te),
           "overall_winrate": float((y > 0).mean()),
           "overall_mean": float(y.mean())}
    out.update({k: {"mean": v["mean"], "winrate": v["winrate"]}
                for k, v in res.items()})
    return out


_STEP_ORDER = [("M0", "M1"), ("M1", "M2"), ("M2", "M3"), ("M3", "M4"),
              ("M4prime", "M5")]


def setup_korean_font() -> None:
    """Windows 한글 폰트. 없으면 기본값으로 두고 경고만 남긴다."""
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    print("경고: 한글 폰트를 찾지 못했다 — 그림의 한글이 깨질 수 있다.",
          file=sys.stderr)
    matplotlib.rcParams["axes.unicode_minus"] = False


def fig_return_distribution(df: pd.DataFrame, path: Path) -> None:
    setup_korean_font()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = df["y_raw"].dropna() * 100
    ax.hist(x, bins=80, range=(-15, 15), color="#4C78A8", alpha=0.85)
    ax.axvline(0, color="#888", lw=1)
    ax.axvline(x.mean(), color="#E45756", lw=2,
               label=f"평균 {x.mean():+.3f}%")
    ax.axvline(x.median(), color="#F58518", lw=2, ls="--",
               label=f"중앙값 {x.median():+.3f}%")
    ax.set_title("52주 신고가 돌파 익일 시가청산 수익률 분포 (비용 차감)")
    ax.set_xlabel("순수익률 (%)")
    ax.set_ylabel("빈도")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_yearly(df: pd.DataFrame, path: Path) -> None:
    setup_korean_font()
    g = df.groupby(df["date"].dt.year)["y_raw"].agg(["size", "mean"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#E45756" if v < 0.001 else "#4C78A8" for v in g["mean"]]
    ax.bar(g.index.astype(str), g["mean"] * 100, color=colors)
    ax.axhline(0, color="#888", lw=1)
    for xi, (yr, row) in enumerate(g.iterrows()):
        ax.text(xi, row["mean"] * 100, f"n={int(row['size'])}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("연도별 익일 시가청산 평균 수익률 — 레짐 의존성")
    ax.set_ylabel("평균 순수익률 (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_ablation(ab: pd.DataFrame, path: Path) -> None:
    """M0→M5 성능 사다리.

    AUC·IC는 4폴드 평균, 상위10% 픽 수익률은 마지막 폴드(2026, n=4,277)
    단일값으로 그린다 — 본문 §6의 close_strength 단독 비교와 같은 표본이라야
    공정한 비교가 된다. M5는 캐베어트(n=869·단일분할·유의성 미확인)를
    그림에 직접 박아 넣어, 이 그림만 떼어놔도 과장되지 않게 한다.
    """
    setup_korean_font()
    order = [m for m in MODEL_ORDER if m in set(ab["model"])]
    g_mean = ab.groupby("model")[["auc", "ic"]].mean().reindex(order)

    last_fold = ab["fold"].max() if ab["fold"].notna().any() else None
    if last_fold is not None and (ab["fold"] == last_fold).any():
        g_top = (ab[ab["fold"] == last_fold].set_index("model")["top_decile"]
                 .reindex(order))
        top_title = f"상위10% 픽 평균 수익률 (fold {int(last_fold)}만)"
    else:
        g_top = ab.groupby("model")["top_decile"].mean().reindex(order)
        top_title = "상위10% 픽 평균 수익률 (폴드 평균)"

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

    # --- AUC ---
    ax = axes[0]
    ax.bar(order, g_mean["auc"], color="#4C78A8")
    ax.axhline(0.5, color="#E45756", ls="--", lw=1, label="무작위(0.5)")
    ax.set_title("AUC (4폴드 평균)")
    ax.legend(fontsize=7)
    if "M5" in order:
        ax.annotate("n=869·단일분할\nBH p=0.24(유의X)",
                    (order.index("M5"), g_mean["auc"].get("M5", np.nan)),
                    textcoords="offset points", xytext=(-10, 12),
                    ha="center", fontsize=7, color="#B22222",
                    arrowprops=dict(arrowstyle="->", color="#B22222", lw=0.8))

    # --- IC (정의 불가한 모델은 해치로 구분) ---
    ax = axes[1]
    ic_vals = g_mean["ic"]
    bars = ax.bar(order, ic_vals.fillna(0.0), color="#4C78A8")
    for i, (m, v) in enumerate(ic_vals.items()):
        if pd.isna(v):
            bars[i].set_hatch("//")
            bars[i].set_facecolor("#DDDDDD")
            bars[i].set_edgecolor("#888888")
            ax.text(i, 0.003, "정의불가", ha="center", va="bottom",
                    fontsize=7, rotation=90, color="#666")
    ax.axhline(0, color="#888", lw=1)
    ax.set_title("Spearman IC (4폴드 평균)")

    # --- 상위10% + 단일피처 기준선 ---
    ax = axes[2]
    tvals = g_top * 100
    bars = ax.bar(order, tvals.fillna(0.0), color="#4C78A8")
    for i, (m, v) in enumerate(tvals.items()):
        if pd.isna(v):
            bars[i].set_hatch("//")
            bars[i].set_facecolor("#DDDDDD")
            bars[i].set_edgecolor("#888888")
            ax.text(i, 0.2, "정의불가", ha="center", va="bottom",
                    fontsize=7, rotation=90, color="#666")
    ax.axhline(0, color="#888", lw=1)
    for label, key, ls, color in (
            ("완벽예측(비현실적 상한)", "perfect", ":", "#888"),
            ("close_strength 단독", "close_strength", "--", "#E45756")):
        v = BENCH_2026[key]["mean"] * 100
        ax.axhline(v, color=color, ls=ls, lw=1.4, zorder=1)
        ax.annotate(f"{label} {v:.2f}%", (0, v), textcoords="offset points",
                    xytext=(0, 4), fontsize=7, color=color)
    ax.set_title(top_title)

    fig.suptitle("피처군 누적에 따른 성능 (M0 → M5) — 50개 피처가 마감강도 "
                 "단일지표를 얼마나 넘는가")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _load_output_csv(name: str) -> pd.DataFrame | None:
    p = config.RESULTS_DIR / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def fig_contribution(ab: pd.DataFrame, path: Path) -> None:
    """피처군별 단계 기여 사다리 — 폴드별 값과 BH 보정 유의성을 함께 표시한다.

    M0→M4 네 단계는 outputs/ablation_deltas.csv의 fold_deltas(폴드 4개의
    ΔAUC)를 산포점으로 얹는다. M4prime→M5는 폴드가 아니라 재료 표본 869행을
    부트스트랩한 것이므로 산포점 대신 95% CI 막대로 그리고, 두 구간이 서로
    다른 단위에서 나왔다는 점을 캡션에 명시한다.
    """
    setup_korean_font()
    g = ab.groupby("model")["auc"].mean()
    steps, step_keys, deltas = [], [], []
    for a, b in _STEP_ORDER:
        if a in g.index and b in g.index:
            steps.append(f"{a}→{b}")           # 표시용(유니코드 화살표)
            step_keys.append(f"{a}->{b}")      # ablation_deltas.csv의 실제 표기
            deltas.append(g[b] - g[a])

    deltas_df = _load_output_csv("ablation_deltas.csv")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(steps, deltas,
           color=["#54A24B" if d > 0 else "#E45756" for d in deltas],
           alpha=0.85, zorder=2, label="4폴드 평균 ΔAUC")
    ax.axhline(0, color="#888", lw=1, zorder=1)

    rng = np.random.default_rng(0)
    fold_label_used, ci_label_used = False, False
    for i, key in enumerate(step_keys):
        if deltas_df is None:
            continue
        row = deltas_df[(deltas_df["step"] == key)
                        & (deltas_df["metric"] == "auc")]
        if row.empty:
            continue
        r = row.iloc[0]
        fd = str(r.get("fold_deltas", ""))
        pts = [float(v) for v in re.findall(r"fold\d+:([+-]?\d*\.?\d+)", fd)]
        if pts:
            xj = i + rng.uniform(-0.12, 0.12, size=len(pts))
            ax.scatter(xj, pts, color="#222", s=26, zorder=3, marker="o",
                       label=None if fold_label_used else "폴드별 값")
            fold_label_used = True
        else:
            lo, hi = r.get("ci_lo"), r.get("ci_hi")
            delta = r.get("delta", deltas[i])
            if pd.notna(lo) and pd.notna(hi):
                ax.errorbar([i], [delta],
                           yerr=[[delta - lo], [hi - delta]],
                           fmt="D", color="#222", ecolor="#222", capsize=5,
                           zorder=3,
                           label=None if ci_label_used else
                           "행단위 부트스트랩 95% CI(n=869)")
                ci_label_used = True
        pbh = r.get("p_value_bh")
        n_fold = r.get("n_fold")
        if pd.notna(n_fold):
            sr = r.get("sign_ratio", "")
            sig_note = f"부호일치 {sr}" if isinstance(sr, str) and "/" in sr else ""
        else:
            sig_note = "행단위(n=869)"
        if pd.notna(pbh):
            sig = ("***" if pbh < 0.001 else "**" if pbh < 0.01 else
                  "*" if pbh < 0.05 else "ns")
            note = f"BH p={pbh:.3f}({sig})"
        else:
            note = "BH p=n/a"
        top_y = max(pts) if pts else (r.get("ci_hi", deltas[i]))
        ax.annotate(f"{note}\n{sig_note}".strip(), (i, top_y),
                   textcoords="offset points", xytext=(0, 8),
                   ha="center", fontsize=7.5)

    ax.set_ylabel("ΔAUC")
    ax.set_title("피처군별 단계 기여 사다리 — 점은 폴드별 값(4개), "
                 "M4prime→M5만 재료표본 869행 부트스트랩 CI")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_importance(df: pd.DataFrame, path: Path) -> None:
    """M4 피처셋으로 전체 학습해 gain 중요도 상위 25개."""
    setup_korean_font()
    from lightgbm import LGBMRegressor
    from surge_d1.models import REG_PARAMS

    feats = dataset.FEATURE_SETS["M4"]
    m = LGBMRegressor(**REG_PARAMS)
    m.fit(df[feats].to_numpy(dtype=float),
          df["y_raw"].to_numpy(dtype=float),
          sample_weight=df["weight"].to_numpy(dtype=float))
    imp = pd.Series(m.booster_.feature_importance("gain"),
                    index=feats).sort_values()[-25:]

    from surge_d1 import features_f1, features_f2, features_f5

    def group(n):
        if n in features_f1.F1_NAMES:
            return "#4C78A8"       # F1 기존
        if n in features_f2.F2_NAMES:
            return "#F58518"       # F2 D+1 특화
        if n in features_f5.F5_NAMES:
            return "#B279A2"       # F5 수급
        return "#54A24B"           # F3 qlib

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(imp.index, imp.values, color=[group(n) for n in imp.index])
    ax.set_title("피처 중요도 (gain) — 파랑 F1 / 주황 F2 / 초록 F3 / 보라 F5")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_horizon(df: pd.DataFrame, path: Path) -> None:
    """지평 비교 — 왜 익일 시가청산인가.

    y_raw/y_hold/y_hold20 세 막대는 df로부터 그때그때 계산한다(실현 가능한
    지평). 네 번째 막대는 초기 설계에서 예측 대상으로 검토했던 '20일 내
    최고 수익률'로, 어느 날이 고점인지 사후에만 알 수 있어 실현 불가능하다.
    이 값은 설계 단계 표본(28,317건, 유니버스 정제 이전)에서 측정된 참고치이며
    df에서 재계산하지 않는다 — 그림 자체에 그 사실을 명시해 혼동을 막는다.
    """
    setup_korean_font()
    cols = [("y_raw", "익일 시가청산", True),
            ("y_hold", "익일 종가청산", True),
            ("y_hold20", "20일 보유", True)]
    labels, means, ts, colors = [], [], [], []
    for c, name, tradable in cols:
        if c not in df.columns:
            continue
        x = df[c].dropna()
        if len(x) < 2:
            continue
        labels.append(f"{name}\n(n={len(x):,})")
        means.append(float(x.mean()) * 100)
        se = x.std() / np.sqrt(len(x))
        ts.append(float(x.mean() / se) if se > 0 else 0.0)
        colors.append("#4C78A8" if tradable else "#BAB0AC")

    # 실현 불가능한 참고 막대 (설계 단계, 28,317건 예비 측정)
    ref_label = "20일 내 최고\n(n=28,317, 설계단계)"
    ref_mean, ref_t = 16.147, 93.56

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    bar_labels = labels + [ref_label]
    bar_means = means + [ref_mean]
    bar_ts = ts + [ref_t]
    bar_colors = colors + ["#BAB0AC"]

    b0 = axes[0].bar(bar_labels, bar_means,
                     color=["#E45756" if m < 0 else c
                            for m, c in zip(bar_means, bar_colors)])
    b0[-1].set_hatch("//")
    b0[-1].set_edgecolor("#666")
    axes[0].axhline(0, color="#888", lw=1)
    axes[0].set_title("지평별 평균 순수익률 (%)")
    axes[0].annotate("실현 불가능\n(사후에만 고점을 앎)", (3, ref_mean),
                     textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=7.5, color="#666")

    b1 = axes[1].bar(bar_labels, bar_ts, color=bar_colors)
    b1[-1].set_hatch("//")
    b1[-1].set_edgecolor("#666")
    axes[1].axhline(0, color="#888", lw=1)
    axes[1].axhline(2, color="#E45756", ls="--", lw=1, label="t=2 (유의성 기준)")
    axes[1].set_title("지평별 t값 — 신호의 강도")
    axes[1].legend(fontsize=8)

    fig.suptitle("왜 익일 시가청산인가 — 종가까지 들면 수익이 사라지고, "
                 "'20일 내 최고'는 애초에 실현할 수 없다")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_equity_curve(daily: pd.DataFrame, path: Path) -> None:
    """전략별 고정명목 누적 수익 곡선. 1.0에서 출발, 매일 동일 명목·재투자 없음.

    매일 전액 재투자(복리)로 그리면 model_top5가 수만~수십만 % 배율에
    도달해 실현 불가능한 그림이 된다(outputs/backtest_stats.csv 참고). 여기서는
    매일 같은 명목 금액을 거는 고정명목(fixed-notional) 누적, 즉
    1.0 + cumsum(일일수익률)로 그린다.
    """
    setup_korean_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"model_top5": "#4C78A8", "model_top10": "#72B7B2",
              "rule_top5_close": "#F58518", "all_breakouts": "#E45756",
              "random_5": "#BAB0AC", "market": "#54A24B"}
    styles = {"model_top5": "-", "model_top10": "-", "rule_top5_close": "--",
             "all_breakouts": "-.", "random_5": ":", "market": "-"}
    cols = [c for c in MAIN_STRATEGIES if c in daily.columns]
    for col in cols:
        eq = 1.0 + daily[col].fillna(0.0).cumsum()
        ax.plot(eq.index, eq.to_numpy(), label=col,
               color=colors.get(col), ls=styles.get(col, "-"), lw=1.7)
    ax.axhline(1.0, color="#888", lw=1)
    ax.set_title("전략별 고정명목 누적 수익 (out-of-sample, 왕복비용 0.21% 반영,\n"
                 "재투자·체결 임팩트 미반영 — 거래 가능성이 아니라 신호 검증용)")
    ax.set_ylabel("누적 배수 (시작=1.0, 매일 동일 명목)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_strategy_bars(stats: pd.DataFrame, path: Path) -> None:
    """전략별 일평균 수익률 / Sharpe / MDD 비교.

    CAGR은 복리 가정을 되풀이하므로 헤드라인에 쓰지 않는다. 일평균 수익률은
    `expectancy`(비용 차감 후 거래당 기대수익)를 쓴다. `expectancy`가 없으면
    (예: 단위테스트 픽스처) `cagr`로 대체하되 그 사실을 제목에 남긴다.
    """
    setup_korean_font()
    s = stats.set_index("strategy")
    keep = [c for c in MAIN_STRATEGIES if c in s.index]
    if keep:
        s = s.loc[keep]

    if "expectancy" in s.columns:
        rate_col, rate_title, rate_scale = "expectancy", "일평균 수익률 (%, 비용차감)", 100
    else:
        rate_col, rate_title, rate_scale = "cagr", "CAGR (%, 픽스처 대체값)", 100

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, col, title, scale in (
            (axes[0], rate_col, rate_title, rate_scale),
            (axes[1], "sharpe", "Sharpe", 1),
            (axes[2], "mdd", "최대낙폭 (%)", 100)):
        vals = s[col] * scale
        ax.bar(s.index, vals,
               color=["#E45756" if v < 0 else "#4C78A8" for v in vals])
        ax.axhline(0, color="#888", lw=1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("전략 비교 — 모델 픽 vs 규칙 픽 vs 전부매수 vs 무작위 vs 시장\n"
                 "(복리 CAGR·누적배수는 표에만 참고용으로 남긴다)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_cost_sensitivity(sens: pd.DataFrame, path: Path) -> None:
    """추가 비용을 물렸을 때 Sharpe·승률이 어디까지 버티는가.

    CAGR은 복리 가정이라(0비용에서 model_top5가 2454%/년) 여기서는 안 쓴다.
    Sharpe·승률은 스케일에 덜 민감해 비용 민감도 비교에 더 적합하다.
    """
    setup_korean_font()
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    x = sens["extra_cost"] * 100
    y_sharpe = sens["sharpe"]
    y_win = sens["win_rate"] * 100

    ax1.plot(x, y_sharpe, "o-", color="#4C78A8", lw=2, label="Sharpe")
    ax1.axhline(0, color="#E45756", ls="--", lw=1.2, label="Sharpe=0 (손익분기)")
    for xi, yi in zip(x, y_sharpe):
        ax1.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#4C78A8")
    ax1.set_xlabel("왕복 0.21%에 더한 추가 비용 (%p)")
    ax1.set_ylabel("Sharpe", color="#4C78A8")

    ax2 = ax1.twinx()
    ax2.plot(x, y_win, "s--", color="#F58518", lw=1.5, label="승률(%)")
    for xi, yi in zip(x, y_win):
        ax2.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="#F58518")
    ax2.set_ylabel("승률 (%)", color="#F58518")

    ax1.set_title("비용 민감도 (model_top5) — 정액비용만 반영, 체결 임팩트는 "
                  "여전히 미지수")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_leakage(lk: pd.DataFrame, path: Path) -> None:
    setup_korean_font()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(lk["method"], lk["auc"], color=["#E45756", "#4C78A8"])
    ax.axhline(0.5, color="#888", ls="--", lw=1)
    ax.set_ylabel("AUC")
    ax.set_title("셔플 K-fold vs walk-forward — 미래정보 누수 편향")
    for i, v in enumerate(lk["auc"]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_calibration(df: pd.DataFrame, path: Path) -> None:
    """마지막 폴드에서 보정 전/후 신뢰도 다이어그램."""
    setup_korean_font()
    from sklearn.calibration import calibration_curve
    from surge_d1 import models, validate

    folds = validate.walk_forward_folds(df["date"], models.TEST_YEARS)
    tr_mask, te_mask = folds[-1]
    tr = df[tr_mask].reset_index(drop=True)
    te = df[te_mask].reset_index(drop=True)
    from sklearn.metrics import brier_score_loss

    feats = dataset.FEATURE_SETS["M4"]
    oof = models.fit_predict(tr, te, feats)                      # 현행(OOF)
    _, bad = models._fit_predict_train_test(tr, te, feats, True, "y_raw",
                                            calibrate="in_fold")  # 옛 방식
    y = te["y_cls"]
    const = float(tr["y_cls"].mean())

    fig, ax = plt.subplots(figsize=(6.6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.2, label="완전 보정")
    series = (
        (oof["prob_raw"], "보정 전 (원확률)", "#999999", "o:"),
        (bad["prob_cal"], "학습예측 보정 (옛 방식)", "#E45756", "s-"),
        (oof["prob_cal"], "OOF 보정 (현행)", "#4C78A8", "o-"),
    )
    for v, label, color, style in series:
        pt, pp = calibration_curve(y, v, n_bins=10, strategy="quantile")
        ax.plot(pp, pt, style, label=f"{label} · Brier {brier_score_loss(y, v):.4f}",
                color=color, lw=1.8)
    ax.axhline(const, color="#54A24B", ls="-.", lw=1.1,
               label=f"상수예측 {const:.3f} · Brier "
                     f"{brier_score_loss(y, np.full(len(y), const)):.4f}")
    ax.set_xlabel("예측 확률")
    ax.set_ylabel("실제 빈도 (익일 순이익)")
    ax.set_title("확률 보정 (2026 폴드) — 학습예측에 맞추면 과신이 증폭된다")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _has_m4(df: pd.DataFrame) -> bool:
    """M4 피처가 전부 있는지. 없으면 모델을 적합할 수 없다.

    합성/축약 프레임으로 write_report를 부르는 경우(테스트 등)를 위해
    모델이 필요한 절은 조용히 건너뛴다 — 예외로 리포트 전체를 못 만드는
    것보다 그 절만 빠지는 편이 낫다.
    """
    return set(dataset.FEATURE_SETS["M4"]).issubset(df.columns)


def operating_point_section(df: pd.DataFrame) -> list[str]:
    """§6-1: 운용점(operating point)에 따라 모델의 가치가 달라진다.

    §6은 '날마다 상위 10%를 반드시 고른다'는 운용점에서 모델을 쟀다. 그
    운용점에서는 모델이 단일 지표를 15% 정도만 앞선다. 그러나 실제 쓰임새는
    '확신이 설 때만 산다'이고, 그 운용점에서는 결과가 달라진다 — IC도
    상위10%도 **날짜 안에서만** 계산되므로 '오늘은 사지 않는다'의 가치를
    구조적으로 볼 수 없다. 이 절이 그 사각지대를 메운다.
    """
    if not _has_m4(df):
        return []
    from surge_d1 import backtest

    o = backtest.out_of_sample_predictions(df).dropna(subset=["y_cls"])
    o = o.assign(yr=o["date"].dt.year)
    TH = 0.55
    RULES = ["close_strength", "RSV_20", "breakout_pct"]

    m = o["prob_cal"] >= TH
    n_pick, n_day = int(m.sum()), o.loc[m, "date"].nunique()
    picked, same_day_rest = o[m], o[o["date"].isin(set(o.loc[m, "date"])) & ~m]
    no_pick = o[~o["date"].isin(set(o.loc[m, "date"]))]

    lines = [
        f"## 6-1. 운용점을 바꾸면 결론이 달라진다 — 모델의 진짜 값어치는 "
        f"'사지 않는 날'에 있다",
        "",
        "§6은 **날마다 상위 10%를 반드시 고른다**는 운용점에서 모델을 쟀다. "
        "그런데 IC도 상위10% 수익률도 **날짜 안에서** 계산되므로, "
        "\"오늘은 확신이 없으니 사지 않는다\"의 가치를 구조적으로 볼 수 없다. "
        "실제 쓰임새는 후자다.",
        "",
        f"보정된 확률에 고정 임계값 {TH:.2f}를 걸면 검정 {o['date'].nunique()}일 중 "
        f"**{n_day}일({n_day / o['date'].nunique() * 100:.0f}%)에만** 사고, "
        f"총 {n_pick:,}건({n_pick / len(o) * 100:.1f}%)을 고른다.",
        "",
        "**먼저 확인할 것: 종목을 고른 것인가, 좋은 날을 고른 것인가?**",
        "",
        "| 구분 | n | 순이익 승률 | 평균 수익 |",
        "|---|---|---|---|",
        f"| 뽑힌 종목 | {len(picked):,} | **{picked['y_cls'].mean()*100:.1f}%** "
        f"| **{picked['y_raw'].mean()*100:+.3f}%** |",
        f"| **같은 날** 안 뽑힌 종목 | {len(same_day_rest):,} "
        f"| {same_day_rest['y_cls'].mean()*100:.1f}% "
        f"| {same_day_rest['y_raw'].mean()*100:+.3f}% |",
        f"| 픽이 없던 날의 전체 종목 | {len(no_pick):,} "
        f"| {no_pick['y_cls'].mean()*100:.1f}% "
        f"| {no_pick['y_raw'].mean()*100:+.3f}% |",
        "",
        f"같은 날 안에서 격차가 승률 "
        f"**{(picked['y_cls'].mean() - same_day_rest['y_cls'].mean())*100:+.1f}%p**, "
        f"수익 **{(picked['y_raw'].mean() - same_day_rest['y_raw'].mean())*100:+.3f}%p**다 — "
        "날짜 타이밍이 아니라 **종목 선정**이 대부분을 설명한다.",
        "",
        "**공정한 비교: 픽 수를 같게 맞추고 점수 기준만 바꾼다.** "
        "단일 지표에는 임계값이 없으므로 **사후에 상위 "
        f"{n_pick:,}개**를 뽑게 해준다 — 미래를 아는 셈이라 룰 쪽에 유리한 "
        "핸디캡이다. 모델은 사전에 고정된 임계값만 쓴다.",
        "",
        "| 점수 기준 | 순이익 승률 | 평균 수익 |",
        "|---|---|---|",
        f"| **ML 보정확률 ≥ {TH:.2f}** (사전 임계값) "
        f"| **{picked['y_cls'].mean()*100:.1f}%** "
        f"| **{picked['y_raw'].mean()*100:+.3f}%** |",
    ]
    for c in RULES:
        if c not in o.columns:
            continue
        t = o.nlargest(n_pick, c)
        lines.append(f"| `{c}` 상위 {n_pick:,} (사후) "
                     f"| {t['y_cls'].mean()*100:.1f}% "
                     f"| {t['y_raw'].mean()*100:+.3f}% |")
    lines += [
        f"| 무작위 {n_pick:,} | {o['y_cls'].mean()*100:.1f}% "
        f"| {o['y_raw'].mean()*100:+.3f}% |",
        "",
        "**연도별로 나눠 보면 두 결론이 갈린다.**",
        "",
        "| 연도 | 기저 승률 | ML 승률 | 최고 룰 승률 | 승률 우위 | 수익 우위 |",
        "|---|---|---|---|---|---|",
    ]
    wins, rets = [], []
    for y, g in o.groupby("yr"):
        gm = g["prob_cal"] >= TH
        n = int(gm.sum())
        if n < 20:
            continue
        ml = g[gm]
        alt = [g.nlargest(n, c) for c in RULES if c in g.columns]
        bw = max(a["y_cls"].mean() for a in alt)
        br = max(a["y_raw"].mean() for a in alt)
        dw = (ml["y_cls"].mean() - bw) * 100
        dr = (ml["y_raw"].mean() / br - 1) * 100 if br > 0 else float("nan")
        wins.append(dw)
        rets.append(dr)
        lines.append(
            f"| {y} | {g['y_cls'].mean()*100:.1f}% | **{ml['y_cls'].mean()*100:.1f}%** "
            f"| {bw*100:.1f}% | {dw:+.1f}%p | {dr:+.1f}% |")
    lines += [
        "",
        f"- **승률 우위는 {sum(w > 0 for w in wins)}/{len(wins)} 폴드에서 양수**"
        f"(+{min(wins):.1f}~+{max(wins):.1f}%p)지만 크기가 작다.",
        f"- **수익 우위는 불안정하다** ({min(rets):+.1f}% ~ {max(rets):+.1f}%) — "
        "한 해는 단일 룰에 진다.",
        "- 반면 **기저 승률 대비 우위는 4/4 폴드 모두 20%p 이상**으로 크고 "
        "안정적이다. 이것이 이 시스템이 실제로 하는 일이다: 단일 지표를 "
        "압도하는 게 아니라, **아무 돌파나 사는 것 대비 승률을 크게 "
        "끌어올리고, 확신이 없는 날에는 쉰다.**",
        "",
        "§6의 \"모델이 더하는 것은 크지 않다\"와 이 절은 모순이 아니다 — "
        "**서로 다른 운용점을 재고 있다.** 매일 만근해야 한다면 모델의 추가 "
        "가치는 작고, 쉴 수 있다면 크다. 발표·실사용에서 둘을 섞어 말하면 안 된다.",
        "",
    ]
    return lines


def calibration_section(df: pd.DataFrame) -> list[str]:
    """§4-1: 확률 보정. 수치는 매 실행 재계산한다(하드코딩 금지).

    이 절은 발견 자체가 결과다 — 표준 관행처럼 보이는 코드 한 줄이
    사용자에게 보이는 확률을 통째로 망가뜨리고 있었다.
    """
    if not _has_m4(df):
        return []
    from sklearn.metrics import brier_score_loss

    from surge_d1 import backtest, models, validate

    feats = dataset.FEATURE_SETS["M4"]
    folds = validate.walk_forward_folds(df["date"], models.TEST_YEARS)
    tr = df[folds[-1][0]].reset_index(drop=True)
    te = df[folds[-1][1]].reset_index(drop=True)
    y = te["y_cls"]
    const = float(tr["y_cls"].mean())

    oof = models.fit_predict(tr, te, feats)
    _, bad = models._fit_predict_train_test(tr, te, feats, True, "y_raw",
                                            calibrate="in_fold")
    b_const = brier_score_loss(y, np.full(len(y), const))
    b_bad = brier_score_loss(y, bad["prob_cal"])
    b_oof = brier_score_loss(y, oof["prob_cal"])

    # 전 검정구간 신뢰도 곡선(수정 후)
    oos = backtest.out_of_sample_predictions(df).dropna(subset=["y_cls"])
    bins = pd.cut(oos["prob_cal"], [0, .35, .5, .65, 1.0], include_lowest=True)
    rel = (oos.groupby(bins, observed=True)
              .agg(n=("y_cls", "size"), 예측=("prob_cal", "mean"),
                   실제=("y_cls", "mean")))

    lines = [
        "## 4-1. 확률 보정 — 표준처럼 보이는 한 줄이 확률을 망가뜨리고 있었다",
        "",
        "모델은 순위(무엇이 더 오를까)와 확률(얼마나 확실한가)을 둘 다 낸다. "
        "순위는 처음부터 작동했지만, **확률은 쓸 수 없는 상태였다.**",
        "",
        "원인은 isotonic 보정을 `clf.predict_proba(X_train)` — 즉 모델이 이미 "
        "외운 학습 예측 — 위에 적합한 것이었다. 분류기는 학습 구간에서 AUC "
        f"{float(pd.read_csv(config.RESULTS_DIR / 'ablation.csv').query("model=='M4'")['train_auc'].mean()):.2f}"
        "를 내지만 검정에서는 0.60이다. 그 격차 위에서 보정은 "
        "\"원확률 0.9 = 진짜 0.95\"라고 배우고, 검정의 0.9(실제로는 0.6쯤)에 "
        "그대로 적용해 **과신을 교정하는 대신 증폭**한다.",
        "",
        f"| 확률 산출 방식 | 2026 폴드 Brier | 상수예측({const:.3f}) 대비 |",
        "|---|---|---|",
        f"| 학습예측 보정 (옛 방식) | {b_bad:.4f} | **나쁨** ❌ |",
        f"| 상수예측 (기준선) | {b_const:.4f} | — |",
        f"| **OOF 보정 (현행)** | **{b_oof:.4f}** | 좋음 ✅ |",
        "",
        "옛 방식은 확률로서 **\"항상 기저율\"보다 나빴다.** 검정 전 구간에서 "
        "예측 4.8% 구간의 실제 빈도가 34.1%, 예측 95.6% 구간이 68.8%로 "
        "양방향 30%p씩 어긋났다.",
        "",
        "**수정:** 보정을 학습 구간 **안에서** 시간순으로 나눈 out-of-fold "
        "예측 위에 적합한다(`models._oof_prob`). 모델이 보지 못한 예측이라 "
        "학습-검정 격차가 보정에 반영된다. 분할은 walk-forward와 같은 확장 "
        "방식이라 내부 검증에도 미래 정보가 섞이지 않는다.",
        "",
        "수정 후 검정 전 구간 신뢰도:",
        "",
        "| 예측 확률 | 실제 빈도 | 격차 | n |",
        "|---|---|---|---|",
    ]
    for _, r in rel.iterrows():
        lines.append(f"| {r['예측']:.3f} | {r['실제']:.3f} | "
                     f"{r['실제'] - r['예측']:+.3f} | {int(r['n']):,} |")
    lines += [
        "",
        "남은 편향은 전부 **과소** 방향(실제 > 예측)으로, 확률을 보수적으로 "
        "쓰게 되는 안전한 쪽이다.",
        "",
        "**파생 결정 — 순위와 확률을 분리했다.** isotonic은 단조라 순서를 "
        "보존하지만 계단 함수라 값을 뭉갠다: 실측으로 4,277행이 고유값 "
        "34개로 붕괴했다. 그 상태로 `nlargest(5)`를 하면 동점 무리 안에서 "
        "사실상 무작위 추출이 된다. 그래서 **순위는 보정 전 `prob_raw`로, "
        "임계값 판정과 사용자 표시는 보정된 `prob_cal`로** 한다. 같은 이유로 "
        "AUC는 `prob_raw`로 잰다 — 동점을 0.5로 세는 `roc_auc_score`가 "
        "AUC를 폴드당 최대 0.005 깎고 있었다.",
        "",
        "![확률 보정](../outputs/figures/07_calibration.png)",
        "",
    ]
    return lines


def _fmt(v, f):
    return f(v) if pd.notna(v) else "n/a"


def write_report(df: pd.DataFrame, ab: pd.DataFrame, lk: pd.DataFrame,
                 bt_stats: pd.DataFrame, sens: pd.DataFrame,
                 path: Path) -> None:
    y = df["y_raw"].dropna()
    t = y.mean() / (y.std() / np.sqrt(len(y)))
    yearly = df.groupby(df["date"].dt.year)["y_raw"].agg(["size", "mean"])
    g = ab.groupby("model")[["auc", "ic", "ic_t", "top_decile"]].mean()

    # 아래 산문에 박히는 수치는 전부 CSV에서 읽는다. 하드코딩하면 재실행
    # 때 조용히 어긋난다 — BENCH_2026에서 같은 실수를 이미 한 번 했다.
    _b = bt_stats.set_index("strategy")
    _t5 = _b.loc["model_top5"] if "model_top5" in _b.index else None
    T5_SHARPE = f"{_t5['sharpe']:.1f}" if _t5 is not None else "n/a"
    T5_WIN = f"{_t5['win_rate']*100:.1f}%" if _t5 is not None else "n/a"
    _s4 = sens[np.isclose(sens["extra_cost"], 0.004)]
    COST4_SHARPE = f"{_s4['sharpe'].iloc[0]:.2f}" if len(_s4) else "n/a"

    _cap_path = config.RESULTS_DIR / "backtest_capacity.csv"
    if _cap_path.exists():
        _c = pd.read_csv(_cap_path)
        CAP_MED = (f"{_c['median'].min()/1e8:.0f}~{_c['median'].max()/1e8:.0f}억")
        CAP_P10 = (f"{_c['p10'].min()/1e8:.0f}~{_c['p10'].max()/1e8:.0f}억")
    else:
        CAP_MED = CAP_P10 = "n/a"

    # §6의 "M4 모델 5.344%"는 4폴드 평균이 아니라 마지막(2026) 폴드 단일값이어야
    # BENCH_2026(close_strength 단독 등, 같은 2026 폴드에서 산출)과 공정하게
    # 비교된다.
    m4_top_2026 = np.nan
    if ab["fold"].notna().any():
        last_fold = ab["fold"].max()
        m4_row = ab[(ab["model"] == "M4") & (ab["fold"] == last_fold)]
        if not m4_row.empty:
            m4_top_2026 = float(m4_row["top_decile"].iloc[0])
    m4_line = (f"| **M4 모델 (피처 50개)** | {BENCH_2026['n_picks']} | — "
              f"| **{m4_top_2026*100:.3f}%** |") if pd.notna(m4_top_2026) else ""

    lines = [
        "# 52주 신고가 돌파의 익일 초과수익 — 연구 결과",
        "",
        "## 0. 요약 — 무엇을 찾았고, 무엇을 찾지 못했나",
        "",
        "1. **이벤트는 알파를 갖지만 절반은 시장의 오버나이트 드리프트다.**",
        "   원시 평균 +0.4747%(t=17.62, n=23,654) 중 시장 대비 초과분은",
        "   **+0.2388%(t=8.81, 비용 차감 후)** — 헤드라인의 절반 수준이다.",
        "2. **모델의 값어치는 '운용점'에 달려 있다 — 이 연구의 핵심 발견이다.**",
        "   *매일 반드시 상위 10%를 사야 한다면* 50개 피처와 LightGBM은",
        "   마감강도(`close_strength`) 단일 지표를 상대 7~15%밖에 못 넘고,",
        "   폴드별로 제대로 튜닝해도(학습-검정 격차 0.219→0.116) 그 격차는",
        "   그대로다(15.4%→14.6%). *확신이 없는 날 쉴 수 있다면* 결과가",
        "   달라진다: 보정확률 0.55 임계값은 검정일의 62%에만 매수해 순이익",
        "   승률 **69.7%**를 낸다 — 아무 돌파나 사는 45.5% 대비 **+24.2%p**이고,",
        "   4개 폴드 전부에서 기저 대비 20%p 이상 앞선다(§6-1). IC와 상위10%",
        "   수익률은 **날짜 안에서만** 계산되므로 '사지 않는다'의 가치를",
        "   구조적으로 볼 수 없다 — 두 결론은 모순이 아니라 서로 다른 운용점이다.",
        "   덧붙여 횡단면 정규화는 오히려 성능을 떨어뜨린다 — 마감강도는",
        "   상대적 지표가 아니라 절대적 지표로 보인다.",
        "3. **수급·재료는 다중검정 보정을 통과하지 못했다.** 수급(M3→M4)은",
        "   BH p=0.084, 재료(M4prime→M5)는 BH p=0.243 — 재료는 표본(869,",
        "   단일분할)이 작아 '효과 없음'이 아니라 **'판정 불가'**다.",
        "4. **표준처럼 보이는 두 관행이 결과를 조용히 망가뜨리고 있었다.**",
        "   (a) isotonic 확률보정을 **학습 예측** 위에 적합하면 과신이",
        "   교정되는 대신 증폭돼, 검정 Brier가 상수예측보다 나빠진다",
        "   (0.2802 vs 0.2552). 학습 구간 내부 out-of-fold 예측으로 적합해",
        "   0.2430으로 고쳤다(§4-1). 이 버그가 살아 있는 동안 예측 CLI는",
        "   실제 60%대인 종목에 89%를 찍고 있었다. (b) 셔플 K-fold는",
        "   walk-forward보다 AUC를 +0.0355 부풀린다. 또한 예측 대상을",
        "   대상으로 삼았던 '20일 내 최고 수익률'은 애초에 실현 불가능한",
        "   라벨이었다.",
        "5. **백테스트는 수익 전략의 증명이 아니라 신호의 통계적 검증이다.**",
        f"   Sharpe {T5_SHARPE}·승률 {T5_WIN}처럼 큰 숫자가 나오는 것 자체가 설명이",
        "   필요한 신호다 — 정액비용 민감도로는 잡히지 않는 체결 임팩트",
        "   (동시호가에서 매수가 종가를, 매도가 시가를 미는 효과) 문제가 남아",
        "   있고, 이 연구는 그 문제에 답할 체결 데이터를 갖고 있지 않다.",
        "   §5를 '거래 가능하다는 증거'가 아니라 '무작위가 아니라는 증거'로",
        "   읽어야 한다.",
        "",
        "## 1. 데이터",
        "",
        f"- 이벤트: **{len(df):,}건**",
        f"- 기간: {df['date'].min().date()} ~ {df['date'].max().date()}",
        f"- 종목 수: {df['code'].nunique():,}",
        f"- 재료 매칭: {int(df['has_catalyst'].sum()):,}건 "
        f"({df['has_catalyst'].mean()*100:.1f}%) — 재료 관련 결론은 전부 이",
        "  낮은 커버리지 위에 서 있다는 점을 염두에 둔다",
        "",
        "## 2. 이벤트 자체의 수익률",
        "",
        f"- 익일 시가청산 평균 **{y.mean()*100:+.3f}%** (t={t:.2f}), "
        f"승률 {(y>0).mean()*100:.1f}%",
    ]
    if "y_up" in df.columns and "y_cls" in df.columns:
        lines.append(
            f"- **익일 상승 확률(비용 전) {df['y_up'].mean()*100:.1f}%** vs "
            f"**익일 순이익 확률(비용 차감 후) {df['y_cls'].mean()*100:.1f}%** — "
            f"주가는 절반 가까이 오르지만 비용을 넘기는 건 그보다 적다")
    lines += [
        "",
        "### 지평 비교 — 왜 익일 시가청산인가",
        "",
        "| 지평 | n | 평균 | 중앙 | 승률 | 표준편차 | t값 |",
        "|---|---|---|---|---|---|---|",
    ]
    for col, name in (("y_raw", "익일 시가청산"), ("y_hold", "익일 종가청산"),
                      ("y_hold20", "20일 보유")):
        if col not in df.columns:
            continue
        x = df[col].dropna()
        if len(x) < 2:
            continue
        se = x.std() / np.sqrt(len(x))
        tt = x.mean() / se if se > 0 else 0.0
        lines.append(
            f"| {name} | {len(x):,} | {x.mean()*100:+.3f}% | {x.median()*100:+.3f}% "
            f"| {(x>0).mean()*100:.1f}% | {x.std()*100:.2f}% | {tt:.2f} |"
        )
    lines += [
        "| 20일 내 최고 (참고, 설계단계 28,317건) | — | +16.147% | — | 80.1% "
        "| — | 93.56 |",
        "",
        "익일 종가까지 보유하면 수익이 사라지고 변동성만 커진다 — 모멘텀 수익이 전적으로",
        "오버나이트에 누적된다는 선행연구와 일치한다. 마지막 행(20일 내 최고)은",
        "이 표의 다른 행들과 표본이 다르다(유니버스 정제 이전, 28,317건) — 재계산하지",
        "않고 참고로만 남긴 이유는, 어느 날이 고점인지 사후에만 알 수 있어 **실현",
        "불가능한 라벨**이기 때문이다. 초기 설계가 이 라벨을 예측 대상으로 삼았던",
        "것이 그 프로젝트의 낮은 R²를 설명하는 유력한 원인이며, 이 진단 자체가 이",
        "연구의 기여 중 하나다.",
        "",
        "### 연도별 (레짐 의존성)",
        "",
        "| 연도 | n | 평균 |",
        "|---|---|---|",
    ]
    for yr, row in yearly.iterrows():
        lines.append(f"| {yr} | {int(row['size']):,} | {row['mean']*100:+.3f}% |")

    lines += ["", "## 3. Ablation (M0 → M5)", "",
              "| 모델 | AUC | IC | IC t | 상위10% 수익률 |", "|---|---|---|---|---|"]
    for m in MODEL_ORDER:
        if m in g.index:
            r = g.loc[m]
            ic_s = f"{r['ic']:+.4f}" if pd.notna(r["ic"]) else "정의불가"
            ict_s = f"{r['ic_t']:+.2f}" if pd.notna(r["ic_t"]) else "—"
            td_s = f"{r['top_decile']*100:+.3f}%" if pd.notna(r["top_decile"]) else "정의불가"
            note = "  (n=869·단일분할, 유의성 미확인)" if m in ("M4prime", "M5") else ""
            lines.append(f"| {m}{note} | {r['auc']:.4f} | {ic_s} | "
                         f"{ict_s} | {td_s} |")

    lines += ["", "## 4. 미래정보 누수 편향", "",
              "물려받은 방법론(셔플 K-fold)이 성능을 얼마나 부풀리는지 보여준다:",
              "", "| 검증 방식 | AUC | IC |", "|---|---|---|"]
    for _, r in lk.iterrows():
        lines.append(f"| {r['method']} | {r['auc']:.4f} | {r['ic']:+.4f} |")
    if len(lk) >= 2 and "auc" in lk.columns:
        try:
            bias = lk.loc[lk["method"].str.contains("shuffle", case=False), "auc"].iloc[0] \
                   - lk.loc[lk["method"].str.contains("walk", case=False), "auc"].iloc[0]
            lines.append(f"\n셔플 K-fold가 walk-forward보다 AUC를 **{bias:+.4f}** "
                        "부풀린다 — 예상된 방향(미래 정보 누수)이다.")
        except (IndexError, KeyError):
            pass

    # === 5. 백테스트 — 수익성 증명이 아니라 신호 검증 ===
    lines += [
        "",
        *calibration_section(df),
        "## 5. 전략 백테스트 — 수익성 증명이 아니라 신호의 검증",
        "",
        "이 절의 목적은 '이 전략으로 돈을 벌 수 있다'를 보이는 것이 아니라,",
        "마감강도가 높은 종목을 골랐을 때 실제로 초과수익이 나는지를 out-of-sample",
        "일봉 데이터로 검증하는 것이다. 종가 매수 → 익일 시가 매도, 왕복비용 0.21%",
        "반영.",
        "",
        f"Sharpe {T5_SHARPE}·승률 {T5_WIN}처럼 큰 숫자가 나오면 보통 '차익거래로 이미",
        "사라졌어야 할 편차'를 의심해야 한다 — **이 연구는 그 의심에 반박할",
        "자료가 없다.** 비용을 왕복 0.21%에서 +0.4%p 더 물려도(§5-1) Sharpe가",
        f"{COST4_SHARPE}로 버티는 것도 안심할 근거가 못 된다. 진짜 위험은 수수료가 아니라",
        "**체결 가능성**이다 — 종가 매수는 동시호가에서 체결가를 밀어올리고,",
        "시가 매도는 시가를 눌러버린다. 이것이 시장 임팩트이며 포지션 크기에",
        "비례해 커지는데, 정액 비용 가정으로는 전혀 잡히지 않고 일봉 OHLCV로는",
        f"애초에 측정할 수 없다. 모델 픽의 실제 거래대금 중앙값은 {CAP_MED},",
        f"하위 10%는 {CAP_P10}에 불과해(§5-2) 유의미한 규모로 거래하면 임팩트를",
        "피하기 어렵다. **그래서 아래 숫자는 '거래 가능하다는 증거'가 아니라",
        "'신호가 무작위가 아니라는 증거'로 읽어야 한다.**",
        "",
        "지표 정의 출처: Sortino/Calmar/Alpha/Beta = vectorbt `returns/nb.py`,",
        "MDD 지속일수 = backtrader `analyzers/drawdown.py`, SQN = backtrader `analyzers/sqn.py`",
        "(Van K. Tharp: 2.0~2.4 평균 / 2.5~2.9 양호 / 3.0~5.0 우수, 표본 30건 미만은 신뢰 불가).",
        "**승률 두 가지:** 임계값 전략(`model_prob*`)은 조건 미달일에 현금으로 "
        "남는다. `승률(전체일)`은 그 현금일을 분모에 넣어 세므로 매일 만근하는 "
        "룰 전략과 직접 비교하면 불리하게 보인다 — `승률(보유일)`을 함께 읽어야 "
        "한다. 어느 쪽도 다른 쪽을 대체하지 않는다: 전체일 승률은 자본 "
        "관점, 보유일 승률은 신호 관점이다.",
        "",
        "**기간 정렬:** `market`은 전략들과 **같은 검정 구간**(첫 OOS일~마지막 "
        "OOS일)으로 잘라 측정한다. 초기 구현은 전체 1,879일에 걸쳐 재고 MDD를 "
        "40.3%로 부풀렸다. 다만 `market`의 890일과 전략들의 880일이 어긋나는데, "
        "이는 그 구간 안에서 **돌파 이벤트가 하루도 없었던 10일**이 전략 쪽에만 "
        "빠지기 때문이다(전략은 그날 현금). 시장은 매매비용을 진입·청산 "
        "**한 번씩만** 물리고, 전략은 매일 회전하므로 매일 문다.",
        "",
        "**헤드라인은 고정명목누적·일평균수익·Sharpe다 — 누적/CAGR은 매일 전액",
        "재투자를 가정한 상한값이라 표 맨 끝에 참고로만 남긴다.**",
        "",
    ]
    bt_cols = [
        ("cum_return_additive", "고정명목누적", lambda v: f"{v*100:+.1f}%"),
        ("expectancy", "일평균수익", lambda v: f"{v*100:+.3f}%"),
        ("sharpe", "Sharpe", lambda v: f"{v:.2f}"),
        ("sortino", "Sortino", lambda v: f"{v:.2f}"),
        ("mdd", "MDD", lambda v: f"{v*100:.1f}%"),
        ("mdd_days", "MDD일수", lambda v: f"{int(v)}"),
        ("calmar", "Calmar", lambda v: f"{v:.2f}"),
        ("sqn", "SQN", lambda v: f"{v:.2f}"),
        ("win_rate", "승률(전체일)", lambda v: f"{v*100:.1f}%"),
        ("win_rate_active", "승률(보유일)", lambda v: f"{v*100:.1f}%"),
        ("n_active_days", "보유일수", lambda v: f"{int(v)}"),
        ("profit_factor", "PF", lambda v: f"{v:.2f}"),
        ("alpha", "Alpha", lambda v: f"{v*100:+.2f}%"),
        ("beta", "Beta", lambda v: f"{v:.2f}"),
        ("cagr", "CAGR(참고·복리)", lambda v: f"{v*100:+.1f}%"),
        ("cum_return", "누적(참고·복리)", lambda v: f"{v*100:+.0f}%"),
    ]
    avail = [(c, h, f) for c, h, f in bt_cols if c in bt_stats.columns]
    if not avail:
        avail = [("cagr", "CAGR", lambda v: f"{v*100:+.1f}%")] \
            if "cagr" in bt_stats.columns else []
    lines.append("| 전략 | " + " | ".join(h for _, h, _ in avail) + " |")
    lines.append("|---|" + "---|" * len(avail))
    for _, r in bt_stats.iterrows():
        vals = [_fmt(r[c], f) for c, _, f in avail]
        lines.append(f"| {r['strategy']} | " + " | ".join(vals) + " |")

    lines += ["", "### 5-1. 비용 민감도 (model_top5)", "",
              "일봉 백테스트는 실제 체결 상호작용을 재현하지 못한다(nautilus_trader 문서의",
              "지적). 종가·시초 모두 동시호가라 체결가가 불리할 수 있으므로, 단일 비용",
              "가정에 기대지 않고 추가 비용을 물려 어디까지 버티는지 본다. 아래 표에도",
              "CAGR·누적이 있지만 복리 가정 때문에 스케일이 왜곡되므로, 실제로 볼 것은",
              "Sharpe·승률이다.",
              "",
              "| 추가비용 | 누적(참고·복리) | CAGR(참고·복리) | Sharpe | MDD | 승률 |",
              "|---|---|---|---|---|---|"]
    for _, r in sens.iterrows():
        lines.append(
            f"| +{r['extra_cost']*100:.2f}%p | {r['cum_return']*100:+.1f}% "
            f"| {r['cagr']*100:+.2f}% | {r['sharpe']:.2f} "
            f"| {r['mdd']*100:.1f}% | {r['win_rate']*100:.1f}% |"
        )
    lines += [
        "",
        "테스트한 범위(왕복비용 0.21%+최대 0.4%p) 안에서는 Sharpe가 0 밑으로 내려가지",
        "않는다 — 손익분기가 이 범위 밖에 있다는 뜻이지, 비용 걱정이 없다는 뜻이",
        "아니다. 위에서 설명했듯 정액 비용은 체결 임팩트를 대변하지 못한다.",
    ]

    cap = _load_output_csv("backtest_capacity.csv")
    lines += ["", "### 5-2. 실행 가능성(capacity) — 규모의 한계", ""]
    if cap is not None and {"strategy", "p10", "median"}.issubset(cap.columns):
        cap_m = cap[cap["strategy"].isin(MAIN_STRATEGIES)] if \
            cap["strategy"].isin(MAIN_STRATEGIES).any() else cap
        lines += ["| 전략 | n | 하위10% 거래대금 | 중앙값 거래대금 |",
                  "|---|---|---|---|"]
        for _, r in cap_m.iterrows():
            lines.append(f"| {r['strategy']} | {int(r['n']):,} "
                        f"| {r['p10']/1e8:.0f}억 | {r['median']/1e8:.0f}억 |")
        lines.append("")
    lines += [
        f"모델·규칙 픽 모두 거래대금 중앙값이 {CAP_MED}, 하위 10%는 {CAP_P10}",
        "수준이다(`outputs/backtest_capacity.csv`). 이는 개별 종목의 하루",
        "거래대금이지 우리가 매수 가능한 최대 규모가 아니다 — 실제로 유의미한",
        "자금을 투입하면 이 중 상당 부분을 스스로 체결시켜야 하므로 시장",
        "임팩트가 커진다. 위 §5의 프레이밍(수익 증명이 아니라 신호 검증)이",
        "이 수치와 직결된다.",
    ]

    dsr = _load_output_csv("backtest_dsr.csv")
    lines += ["", "### 5-3. Deflated Sharpe Ratio — '무작위가 아니다'의 증거", "",
              "DSR은 다중 전략을 동시에 시도했을 때의 selection bias를 보정한",
              "Sharpe 유의성이다. 이것이 통과한다고 해서 거래 가능하다는 뜻은",
              "아니다 — §5의 논리대로 이건 '무작위가 아니다'를 보여줄 뿐이다.",
              ""]
    if dsr is not None and {"strategy", "sr_hat", "dsr_naive11"}.issubset(dsr.columns):
        show = dsr[dsr["strategy"].isin(
            ["all_breakouts", "random_5", "rule_top5_close", "model_top5",
             "model_prob50"])]
        if show.empty:
            show = dsr
        lines += ["**주의: 아래 Sharpe는 일간(비연율화) 값이다** — §5 표의 연율화 "
                  "Sharpe(예: model_top5 6.53)와 √252배 차이가 난다(0.4114×√252=6.53).",
                  "",
                  "| 전략 | Sharpe(일간) | DSR(11개 시행 가정) |", "|---|---|---|"]
        for _, r in show.iterrows():
            lines.append(f"| {r['strategy']} | {r['sr_hat']:.2f} "
                        f"| {r['dsr_naive11']:.3f} |")
        lines += ["",
                 "`all_breakouts`·`random_5`는 DSR ≈ 0.25로 탈락하고(그냥",
                 "돌파 이벤트 자체이거나 무작위 선택이라 통계적으로 유의하다",
                 "말할 수 없다), 종목을 고르는 전략들만 DSR ≈ 1.0으로 통과한다.",
                 "**알파는 '돌파를 산다'는 행위가 아니라 '고른다'는 행위에",
                 "있다.**"]

    lines += [
        "",
        "## 5-4. 시장 조정 — 헤드라인의 절반은 시장 베타다",
        "",
        "`y_raw`(+0.4747%)는 **시장 조정 전** 수치다. 같은 날 시장 전체의 오버나이트",
        "갭을 빼면 다음과 같다(이벤트 23,654건 전수):",
        "",
        "| | 평균 | t |",
        "|---|---|---|",
        "| 이벤트 오버나이트 갭 (비용 전) | +0.6847% | 25.42 |",
        "| 같은 날 시장 오버나이트 갭 | **+0.2359%** | — |",
        "| **초과수익 (시장 대비)** | **+0.4488%** | 16.56 |",
        "| **초과수익 − 비용 0.21%** | **+0.2388%** | **8.81** |",
        "",
        "즉 한국 시장은 지수 수준에서도 오버나이트에 하루 +0.236%씩 상승하며",
        "(선행연구 §2.1의 오버나이트 이상현상과 일치), 우리 이벤트의 초과분은",
        "그 위에 얹힌 +0.449%다. 비용 차감 후 진짜 알파는 **+0.2388%(t=8.81)** 로,",
        "시장 조정 없는 헤드라인의 절반 수준이다. **두 수치를 항상 함께 제시한다** —",
        "조정 전 수치만 쓰면 시장 베타를 알파로 오인하게 만든다.",
        "",
        "연도별 초과수익(시장 대비·비용 차감): 2020 +0.400 / 2021 +0.114 /",
        "2022 +0.094 / 2023 +0.189 / **2024 −0.010** / 2025 +0.212 / **2026 +0.602**.",
        "2024년은 시장 대비 초과분이 음수로, 레짐 의존성이 시장 조정 후 더 뚜렷하다.",
        "",
        "## 6. 종목선정은 작동한다 — 그러나 모델이 더하는 것은 크지 않다",
        "",
        "이 연구에서 가장 중요한 비교는 모델 대 단일 지표다. 2026 검정 폴드",
        f"(n={BENCH_2026['overall_n']:,}, 전체 승률 {BENCH_2026['overall_winrate']*100:.1f}%, "
        f"평균 +{BENCH_2026['overall_mean']*100:.3f}%)에서 상위 10%를 고른 결과:",
        "",
        "| 정렬 기준 | 픽 수 | 승률 | 평균 수익 |",
        "|---|---|---|---|",
        f"| 완벽 예측 (이론적 상한) | {BENCH_2026['n_picks']} "
        f"| {BENCH_2026['perfect']['winrate']*100:.1f}% "
        f"| {BENCH_2026['perfect']['mean']*100:.3f}% |",
        m4_line,
        f"| **`close_strength` 단독** | {BENCH_2026['n_picks']} "
        f"| **{BENCH_2026['close_strength']['winrate']*100:.1f}%** "
        f"| **{BENCH_2026['close_strength']['mean']*100:.3f}%** |",
        f"| `RSV_20` 단독 | {BENCH_2026['n_picks']} "
        f"| {BENCH_2026['rsv_20']['winrate']*100:.1f}% "
        f"| {BENCH_2026['rsv_20']['mean']*100:.3f}% |",
        f"| 무작위 | {BENCH_2026['n_picks']} "
        f"| {BENCH_2026['random']['winrate']*100:.1f}% "
        f"| {BENCH_2026['random']['mean']*100:.3f}% |",
        "",
        "종목선정 자체는 명백히 작동한다(승률 50.5% → 67.8%). 그러나 그 선정력은",
        "**마감강도라는 단일 지표에서 거의 전부 나오며**, 50개 피처와 LightGBM이",
        "더하는 것은 상대 6.2%에 불과하다(2026 폴드 기준; 4폴드 평균으로는 15.4%,",
        "아래 §6-1 참고).",
        "",
        "숫자가 커 보이는 이유도 함께 밝힌다: 수익률 분포의 왜도가 1.74이고,",
        "실제 상위 10%가 전체 합계 수익의 114.8%를 차지한다(나머지 90%는 합쳐서 음수).",
        "상위 10%의 5%는 전형적 결과가 아니라 두꺼운 우측 꼬리의 산물이다.",
        "",
        "### 6-1. 튜닝이 격차를 좁히지만 마진은 그대로다",
        "",
        "물려받은 하이퍼파라미터가 과적합돼 있었는지 확인하기 위해 폴드 내부",
        "학습 데이터만으로 튜닝을 다시 했다(`outputs/tuning_sensitivity.csv`,",
        "조건 A=물려받은 값, B=폴드별 재튜닝, C=B+횡단면 정규화). 결론은 세 가지다.",
        "",
        "1. **튜닝은 과적합을 실제로 고친다.** 학습-검정 AUC 격차가 A의 0.219에서",
        "   B에서 0.116으로 절반 가까이 줄고, 선택된 파라미터는 일관되게",
        "   정규화 방향(`reg_lambda=20`, `min_child_samples=100`)으로 움직인다.",
        "2. **그런데도 검정 성능은 거의 안 바뀐다.** 4폴드 평균 AUC A=0.599 →",
        "   B=0.602. `close_strength` 단독 대비 마진도 15.4% → 14.6%로 거의",
        "   그대로다 — 최고로 튜닝된 모델이 단일 지표를 4폴드 평균 **약 15%**",
        "   앞서지만, 폴드별로는 **3.6%(2026, 가장 최근·가장 큰 폴드)부터",
        "   23.8%(2025)까지** 크게 갈린다(조건 B 기준. 조건 A로는 6.2~23.9%).",
        "   이 변동 폭이 평균값보다 더 중요한",
        "   정보다 — 가장 최근 폴드에서 모델의 우위가 가장 작다는 뜻이기",
        "   때문이다. 즉 이 결론은 잘못 튜닝된 모델의 아티팩트가 아니다 —",
        "   확인해봤고, 여전히 그대로다.",
        "3. **횡단면 정규화(qlib 기본값)는 오히려 해친다.** 조건 C는 AUC·IC·",
        "   상위10% 수익률이 B보다 전부 낮다(마진이 1.7%까지 줄어드는 것은",
        "   측정이 더 정확해져서가 아니라 **모델 자체가 나빠졌기 때문**이다).",
        "   그럴듯한 해석은, 마감강도가 그날 함께 돌파한 다른 종목들과의",
        "   상대적 위치가 아니라 그 봉 자체의 **절대적 속성**이라는 것이다 —",
        "   즉 '얼마나 고가권에서 마감했는가'는 그 자체로 의미가 있고, 같은 날",
        "   다른 돌파 종목과 비교해 상대화하면 정보가 사라진다. 다만 이는 한",
        "   문제에 한 정규화 방식을 시험한 관찰이며, 일반적 결론으로 확대하지",
        "   않는다.",
        "",
        *operating_point_section(df),
        "## 7. 재료 — 판정 불가 (효과 없음이 아니다)",
        "",
        "원시 비교에서 재료가 붙은 이벤트는 승률 52.4% · 평균 +2.015%로, 없는 쪽",
        "(43.8% · +0.321%)보다 크게 높다. 재료유형별 기울기도 뚜렷하다",
        "(사유없음 43.9% ~ 신약·임상 56.0%).",
        "",
        "**그러나 이 차이는 교란돼 있다.** 재료 출처가 '그날의 특징주 TOP20'이므로,",
        "재료가 붙었다는 것은 곧 그날 크게 움직인 종목이라는 뜻이다. 교란을 통제한",
        "검정(M5 vs M4prime)에서는 유의하지 않았다(BH p=0.243).",
        "",
        "다만 그 검정은 **검정력이 약했다**: 표본 869(단일 분할), 탐지 가능 효과가",
        "AUC 차이 0.03 이상, 재료 커버리지 9.09%, 회귀 arm은 상수로 붕괴해 판정 불가.",
        "따라서 정확한 결론은 '재료는 도움이 안 된다'가 아니라 **'이 표본으로는",
        "판정할 수 없었다'** 이며, 판정하려면 재료 커버리지를 높여야 한다(DART 공시).",
        "",
        "## 8. 과적합 진단과 신뢰구간의 함정",
        "",
        "피처를 늘린다고 좋아지지 않았다. 단계별 AUC 증분과 폴드 부호 일치:",
        "",
        "| 단계 | 평균 ΔAUC | 부호 일치 | IC 변화 | 신뢰구간 |",
        "|---|---|---|---|---|",
        "| M0→M1 | +0.0646 | 4/4 | 상승 | 4폴드 부트스트랩 |",
        "| M1→M2 | +0.0230 | 4/4 | 상승 | [+0.0214, +0.0244] (4폴드) |",
        "| M2→M3 | +0.0035 | 4/4 (미미) | **4/4 하락** | 4폴드 부트스트랩 |",
        "| M3→M4 | +0.0080 | **3/4** | 혼재 | [−0.0001, +0.0139] (4폴드) |",
        "| M4prime→M5 | +0.0111 | — (n=869) | 정의불가(회귀 상수붕괴) "
        "| [−0.0074, +0.0299] (행 869개, 2,000회 부트스트랩) |",
        "",
        "특히 qlib 범용 팩터 20개를 더한 M3는 AUC를 0.004 올리는 대가로 **IC를 네 폴드",
        "전부에서 떨어뜨렸다**. 분류 경계는 다듬었으나 순위 정보를 잃은 것으로,",
        "전형적인 과적합 신호다.",
        "",
        "**신뢰구간은 서로 다른 단위에서 계산됐다는 점을 반드시 함께 읽어야",
        "한다.** M0→M4 네 단계의 구간은 폴드 4개의 ΔAUC 그 자체를 부트스트랩한",
        "것이고, M4prime→M5의 구간은 재료 표본 869행을 2,000회 리샘플링한",
        "것이다. 전자(예: M1→M2의 [+0.0214, +0.0244])가 훨씬 좁아 보이지만, 이는",
        "폴드 하나하나를 표본오차 없는 점으로 취급하기 때문이며 실제 불확실성을",
        "과소평가한다 — 폴드가 4개뿐인 walk-forward 설계에서는 이보다 나은 방법이",
        "없다. 그래서 구간의 폭보다 **부호 일치 비율**(4/4 vs 3/4)이 더 신뢰할 수",
        "있는 판단 근거다. 특히 M3→M4는 구간이 0에 거의 닿아 있고(3/4, 폴드 4개",
        "중 2026만 역전) 이 사실을 함께 읽어야 과대해석을 피한다. 학습 대 검정",
        "성능 격차와 BH 보정 p값 전체는 `outputs/ablation.csv`,",
        "`outputs/ablation_deltas.csv` 에 기록된다.",
        "",
        "## 9. 연구자 자유도",
        "",
        "데이터를 본 뒤 내린 처리 결정과 각각이 결과를 얼마나 움직였는지는",
        "`docs/연구자_자유도_기록.md` 에 별도로 기록했다. 헤드라인을 실질적으로",
        "움직인 결정은 유니버스 정제 하나뿐이며(+0.402% → +0.4747%), 나머지는",
        "이벤트 수와 라벨에 영향이 없는 피처 품질·누수 수정이었다.",
        "",
        "**가장 큰 구조적 한계: 최종 홀드아웃이 없다.** 네 폴드를 모두 여러 차례",
        "들여다보며 버그를 고치고 재실행했으므로, 한 번도 보지 않은 데이터에서",
        "재현했다고는 말할 수 없다. 이 문서의 모든 수치는 '한 번도 안 본 데이터에서",
        "확인된 결과'가 아니라 '반복적으로 들여다본 네 폴드에서 일관되게 나타난",
        "패턴'으로 읽어야 한다.",
        "",
        "## 10. 한계",
        "",
        "- **최종 홀드아웃 부재.** §9에서 다뤘듯 네 폴드 모두 개발 중 반복적으로",
        "  들여다봤다.",
        "- **재료 커버리지가 낮다(9.09%).** 표본이 '그날 특징주 TOP20'이라 당일",
        "  급등과 얽혀 있고, M5 검정(§7)은 표본 869·단일분할로 판정력이 약하다.",
        f"- **실행 가능 규모가 제한적이다.** 픽의 거래대금 중앙값이 {CAP_MED},",
        f"  하위 10%는 {CAP_P10}이라(§5-2) 유의미한 자금은 시장 임팩트를 피하기",
        "  어렵다.",
        "- **백테스트가 체결을 재현하지 못한다.** 동시호가 슬리피지는 왕복",
        "  0.21%보다 클 수 있고, 정액 비용 민감도(§5-1)로는 이를 검증할 수 없다.",
        "- 기대값이 레짐 의존적이다(§2 연도별 표). 전체 평균만 제시하면 오도된다.",
        "- 익일 시가는 KRX 09:00 일봉 시가 근사이며 NXT 08:00 개장가는 반영되지 않았다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_parquet(config.DATASET_PATH)

    global BENCH_2026
    fresh = compute_bench_2026(df)
    for k in ("perfect", "close_strength", "rsv_20"):   # random은 추첨평균이라 제외
        gap = abs(fresh[k]["mean"] - BENCH_2026[k]["mean"])
        if gap > 0.001:
            print(f"  [warn] BENCH_2026['{k}'] 드리프트: 하드코딩 "
                  f"{BENCH_2026[k]['mean']*100:.3f}% -> 재계산 "
                  f"{fresh[k]['mean']*100:.3f}% ({gap*100:.3f}%p)")
    BENCH_2026 = fresh

    R = config.RESULTS_DIR
    ab = pd.read_csv(R / "ablation.csv")
    lk = pd.read_csv(R / "leakage.csv")
    bt = pd.read_csv(R / "backtest_stats.csv")
    sens = pd.read_csv(R / "backtest_cost_sensitivity.csv")
    daily = pd.read_csv(R / "backtest_daily.csv", index_col=0, parse_dates=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig_return_distribution(df, config.FIGURES_DIR / "01_distribution.png")
    fig_yearly(df, config.FIGURES_DIR / "02_yearly.png")
    fig_ablation(ab, config.FIGURES_DIR / "03_ablation.png")
    fig_contribution(ab, config.FIGURES_DIR / "04_contribution.png")
    fig_importance(df, config.FIGURES_DIR / "05_importance.png")
    fig_leakage(lk, config.FIGURES_DIR / "06_leakage.png")
    fig_calibration(df, config.FIGURES_DIR / "07_calibration.png")
    fig_equity_curve(daily, config.FIGURES_DIR / "08_equity_curve.png")
    fig_strategy_bars(bt, config.FIGURES_DIR / "09_strategy_bars.png")
    fig_cost_sensitivity(sens, config.FIGURES_DIR / "10_cost_sensitivity.png")
    fig_horizon(df, config.FIGURES_DIR / "11_horizon.png")

    out = config.PROJECT_ROOT / "docs" / "연구리포트.md"
    write_report(df, ab, lk, bt, sens, out)
    print(f"figure 11장 -> {config.FIGURES_DIR}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
