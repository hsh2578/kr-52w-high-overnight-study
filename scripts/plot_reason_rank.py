# -*- coding: utf-8 -*-
"""텍스트 순위 모형 결과 그림 — 상한가를 빼면 순위상관이 사라진다.

입력: outputs/reason_rank_holdout.csv (reason_rank.py --holdout 의 산출물)
출력: outputs/figures/12_reason_rank_ic.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "reason_rank_holdout.csv"
OUT = ROOT / "outputs" / "figures" / "12_reason_rank_ic.png"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = {"전체": ("#2a78d6", "상한가 포함"), "상한가 제외": ("#eb6834", "상한가 제외 (살 수 있는 종목)")}
MODELS = {"price": "가격 기준선\n(등락률·상한가·거래대금)", "emb": "사유 문장 임베딩\n(kf-deberta)"}

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})


def main() -> None:
    d = pd.read_csv(SRC)
    d = d[d["모형"].isin(MODELS)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharex=True, facecolor=SURFACE)
    for ax, (period, sub) in zip(axes, [("개발", "개발 구간 · 2025-07 ~ 2026-08"),
                                        ("홀드아웃", "홀드아웃 · 2026-09 ~ 11 (1회 평가)")]):
        g = d[d["구간"] == period]
        ax.set_facecolor(SURFACE)
        h, gap = 0.26, 0.03
        for k, m in enumerate(MODELS):
            for j, (sample, (color, _)) in enumerate(SERIES.items()):
                r = g[(g["모형"] == m) & (g["표본"] == sample)].iloc[0]
                y = k + (j - 0.5) * (h + gap)
                ax.barh(y, r["ic"], height=h, color=color, zorder=3)
                ax.text(max(r["ic"], 0) + 0.008, y, f"{r['ic']:.3f}  (t={r['ic_t']:.1f})",
                        va="center", ha="left", fontsize=9.5, color=INK)
        ax.set_yticks(range(len(MODELS)), list(MODELS.values()), fontsize=10, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 0.47)
        ax.set_title(sub, loc="left", fontsize=10.5, color=MUTED, pad=8)
        ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
        ax.axvline(0, color=MUTED, linewidth=1, zorder=4)
        ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("일별 순위상관(IC) 평균", fontsize=9.5, color=MUTED)
    axes[1].tick_params(axis="y", labelleft=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in SERIES.values()]
    fig.legend(handles, [n for _, n in SERIES.values()], loc="upper left",
               bbox_to_anchor=(0.012, 0.9), ncol=2, frameon=False, fontsize=10,
               handlelength=1.1, columnspacing=1.6)
    fig.suptitle("상한가를 빼면 순위를 가리지 못한다", x=0.018, y=0.985, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.74, bottom=0.15, wspace=0.08)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    print("저장", OUT)


if __name__ == "__main__":
    main()
