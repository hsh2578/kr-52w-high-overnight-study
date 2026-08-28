# -*- coding: utf-8 -*-
"""전략 백테스트: 모델 픽 vs 전부매수 vs 무작위.

상위 decile 평균 수익률은 자본배분·복리·낙폭을 뺀 숫자라 알파의 효용을 말해주지
못한다. 여기서는 같은 자본으로 굴렸을 때의 누적수익·샤프·MDD를 비교한다.

수익률은 이미 왕복비용(0.21%)이 차감된 y_raw 를 쓴다. 즉 '종가 매수 → 익일 시가 매도'
1회전이 하루 수익률이며, 매일 전액 재투자한다고 가정한다.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from surge_d1 import config

TRADING_DAYS = 252


def simulate(df: pd.DataFrame, score_col: str,
             top_n: int | None) -> pd.DataFrame:
    """날짜별로 score 상위 top_n을 균등 매수한 포트폴리오의 일별 수익률.

    top_n=None 이면 그날 후보 전부를 균등 매수한다.
    후보 수가 top_n보다 적으면 있는 만큼만 산다(현금 보유 없음, 균등 배분).
    """
    rows = []
    for date, g in df.groupby("date"):
        picked = g if top_n is None else g.nlargest(top_n, score_col)
        if len(picked) == 0:
            continue
        rows.append({"date": date,
                     "ret": float(picked["y_raw"].mean()),
                     "n_picked": int(len(picked))})
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame({"ret": [], "n_picked": []},
                            index=pd.to_datetime([]))
    return out.set_index("date").sort_index()


def simulate_random(df: pd.DataFrame, top_n: int, n_iter: int = 20,
                    seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """무작위 top_n 선택을 n_iter회 반복한 평균 일별 수익률(비교군)."""
    rng = np.random.default_rng(seed)
    acc = None
    for _ in range(n_iter):
        rows = []
        for date, g in df.groupby("date"):
            k = min(top_n, len(g))
            if k == 0:
                continue
            idx = rng.choice(len(g), size=k, replace=False)
            rows.append({"date": date,
                         "ret": float(g["y_raw"].to_numpy()[idx].mean()),
                         "n_picked": int(k)})
        cur = pd.DataFrame(rows).set_index("date").sort_index()
        acc = cur if acc is None else acc.add(cur, fill_value=0.0)
    if acc is None:
        return pd.DataFrame({"ret": [], "n_picked": []},
                            index=pd.to_datetime([]))
    acc["ret"] /= n_iter
    acc["n_picked"] /= n_iter
    return acc


def _empty_stats() -> dict:
    keys = ("cum_return cagr vol sharpe sortino calmar mdd mdd_days win_rate "
            "profit_factor expectancy sqn best_day worst_day alpha beta").split()
    out = {k: float("nan") for k in keys}
    out.update({"cum_return": 0.0, "cagr": 0.0, "vol": 0.0, "mdd": 0.0,
                "mdd_days": 0, "n_days": 0, "n_trades": 0})
    return out


def _max_drawdown_days(equity: pd.Series) -> int:
    """최대낙폭이 지속된 기간(거래일).

    backtrader DrawDown 분석기 방식: 고점 대비 낙폭 상태가 이어진 봉 수를 세고
    낙폭이 0이 되면 리셋한다. 그중 최댓값을 돌려준다.
    """
    peak = equity.cummax()
    under = (equity < peak).to_numpy()
    best = cur = 0
    for flag in under:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return int(best)


def performance_stats(daily: pd.DataFrame,
                      benchmark: pd.Series | None = None,
                      risk_free: float = 0.0) -> dict:
    """일별 수익률 → 성과 지표. 매일 전액 재투자(복리) 가정.

    지표 정의는 공개 라이브러리를 그대로 따랐다(자의적 공식이 아님):
      - Sortino / Calmar / Alpha / Beta : vectorbt `vectorbt/returns/nb.py`
      - MDD 지속일수                    : backtrader `analyzers/drawdown.py`
      - SQN                             : backtrader `analyzers/sqn.py` (Van K. Tharp)

    risk_free 는 **연율**로 받아 기하적으로 일할 환산한다(backtrader convertrate 방식):
        일간 무위험수익률 = (1 + 연율)^(1/252) - 1
    기본값 0인 이유: 이 전략은 하룻밤만 시장에 노출되고 나머지 시간은 현금이라
    무위험수익률을 빼는 것이 오히려 과도한 페널티다. 민감도는 리포트에 별도 표기한다.
    """
    r = daily["ret"].astype(float).dropna() if len(daily) else pd.Series(dtype=float)
    n = len(r)
    if n == 0:
        return _empty_stats()

    rf_daily = (1.0 + risk_free) ** (1.0 / TRADING_DAYS) - 1.0
    excess = r - rf_daily

    equity = (1.0 + r).cumprod()
    cum_return = float(equity.iloc[-1] - 1.0)
    # 연환산 수익률: (∏(1+r))^(252/n) - 1   (vectorbt annualized_return)
    cagr = float(equity.iloc[-1] ** (TRADING_DAYS / n) - 1.0)

    sd = float(r.std(ddof=1)) if n > 1 else 0.0     # 표본 표준편차(Bessel)
    vol = sd * np.sqrt(TRADING_DAYS)
    sharpe = (float(excess.mean() / sd * np.sqrt(TRADING_DAYS))
              if sd > 0 else float("nan"))

    # Sortino: 하방편차의 평균은 **전체 관측치**로 나눈다(vectorbt downside_risk)
    downside = np.minimum(excess.to_numpy(), 0.0)
    dd_risk = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS))
    sortino = (float(excess.mean() * TRADING_DAYS / dd_risk)
               if dd_risk > 0 else float("nan"))

    peak = equity.cummax()
    mdd = float((1.0 - equity / peak).max())
    calmar = float(cagr / mdd) if mdd > 0 else float("nan")

    wins, losses = r[r > 0], r[r < 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("nan")

    # SQN = √N × mean/std  (N<30이면 신뢰 불가 → NaN)
    sqn = float(np.sqrt(n) * r.mean() / sd) if (sd > 0 and n >= 30) else float("nan")

    alpha = beta = float("nan")
    if benchmark is not None:
        b = benchmark.reindex(r.index).astype(float)
        mask = b.notna() & r.notna()
        if mask.sum() > 2:
            rr, bb = r[mask].to_numpy(), b[mask].to_numpy()
            bres = bb - bb.mean()
            var_b = float(np.mean(bres ** 2))
            if var_b > 0:
                beta = float(np.mean(bres * rr) / var_b)
                a_series = (rr - rf_daily) - beta * (bb - rf_daily)
                alpha = float((np.mean(a_series) + 1.0) ** TRADING_DAYS - 1.0)

    return {
        "cum_return": cum_return, "cagr": cagr, "vol": vol,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "mdd": mdd, "mdd_days": _max_drawdown_days(equity),
        "win_rate": float((r > 0).mean()),
        "profit_factor": profit_factor,
        "expectancy": float(r.mean()),
        "sqn": sqn,
        "best_day": float(r.max()), "worst_day": float(r.min()),
        "alpha": alpha, "beta": beta,
        "n_days": int(n), "n_trades": int(daily["n_picked"].sum()),
    }


def out_of_sample_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """walk-forward 폴드별 검정 구간 예측만 모아 이어 붙인다.

    학습 구간 예측으로 백테스트하면 전부 거짓이므로 반드시 이 경로를 쓴다.
    """
    from surge_d1 import dataset, models, validate

    feats = dataset.FEATURE_SETS["M4"]
    parts = []
    for tr_mask, te_mask in validate.walk_forward_folds(df["date"],
                                                        models.TEST_YEARS):
        tr = df[tr_mask].reset_index(drop=True)
        te = df[te_mask].reset_index(drop=True)
        out = models.fit_predict(tr, te, feats)
        te = te.copy()
        te["score"] = out["prob_cal"]
        te["pred_ret"] = out["pred_reg"]

        # 분위수 임계값은 **학습 구간 예측**으로 정한다. 검정 구간 예측으로 정하면
        # 결과를 보고 고르는 셈이라 체리피킹이다.
        tr_out = models.fit_predict(tr, tr, feats)
        thr = float(np.quantile(tr_out["prob_cal"], 0.80))
        te["pick_q80"] = (te["score"] >= thr).astype(int)

        parts.append(te)
    if not parts:
        return df.head(0).assign(score=[], pred_ret=[], pick_q80=[])
    return pd.concat(parts, ignore_index=True)


def simulate_threshold(df: pd.DataFrame, score_col: str,
                       threshold: float) -> pd.DataFrame:
    """score가 임계값 이상인 종목만 균등 매수. 조건 미달일이면 현금(수익률 0).

    **현금일을 빼면 안 된다.** 픽이 없는 날을 그냥 건너뛰면 거래일 수가 줄어 CAGR이
    부풀려지고, 자산곡선도 실제로 굴렸을 때와 달라진다. 조건이 까다로워 픽이 드물수록
    이 처리가 결과를 좌우한다.
    """
    rows = []
    for date, g in df.groupby("date"):
        picked = g[g[score_col] >= threshold]
        if len(picked) == 0:
            rows.append({"date": date, "ret": 0.0, "n_picked": 0})
        else:
            rows.append({"date": date, "ret": float(picked["y_raw"].mean()),
                         "n_picked": int(len(picked))})
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame({"ret": [], "n_picked": []},
                            index=pd.to_datetime([]))
    return out.set_index("date").sort_index()


def tick_size(price: float) -> float:
    """KRX 호가단위(2023-01 개정). 슬리피지 1틱을 금액으로 환산할 때 쓴다."""
    for cap, tick in ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50),
                      (200_000, 100), (500_000, 500)):
        if price < cap:
            return float(tick)
    return 1_000.0


def cost_sensitivity(oos: pd.DataFrame, top_n: int = 5,
                     extra_costs=(0.0, 0.001, 0.002, 0.004)) -> pd.DataFrame:
    """비용을 더 물렸을 때 전략이 언제 무너지는지 본다.

    nautilus_trader 문서가 지적하듯, 일봉(OHLC) 백테스트는 실제 체결 상호작용을
    재현하지 못한다. 우리는 종가 매수·익일 시가 매도를 가정하는데 둘 다 동시호가라
    실제 체결가가 불리할 수 있다. 그래서 단일 비용 가정에 기대지 않고, 추가 비용을
    단계적으로 물려 **엣지가 사라지는 지점**을 보고한다.
    """
    rows = []
    for extra in extra_costs:
        adj = oos.copy()
        adj["y_raw"] = adj["y_raw"] - extra
        d = simulate(adj, "score", top_n)
        s = performance_stats(d)
        rows.append({"extra_cost": extra, "cum_return": s["cum_return"],
                     "cagr": s["cagr"], "sharpe": s["sharpe"],
                     "mdd": s["mdd"], "win_rate": s["win_rate"]})
    return pd.DataFrame(rows)


def deflated_sharpe_ratio(daily_ret: pd.Series, n_trials: int,
                          trial_sharpes: np.ndarray | None = None) -> dict:
    """López de Prado 방식 Deflated Sharpe Ratio (DSR).

    출처: Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe
    Ratio: Correcting for Selection Bias, Backtest Overfitting, and
    Non-Normality." Journal of Portfolio Management, 40(5), 94-107.

    여러 전략을 탐색해 그중 최고를 보고하면, 어느 전략도 진짜 스킬이 없어도
    "최댓값"이라는 이유만으로 관측 샤프비가 위로 편향된다(선택 편향). DSR은
    (1) 시행 횟수 n_trials, (2) 표본 길이(=일수), (3) 수익률 분포의 왜도·첨도로
    관측 샤프비를 할인해, "이 샤프비가 순수한 다중검정 우연보다 유의한가"를
    확률로 답한다.

    daily_ret     : 일별 수익률(연환산하지 않은 원계열) — 이 전략 하나의 것.
    n_trials      : 독립 시행 횟수로 간주할 값. 서로 다른 전략이 사실상 같은
                   신호의 변형(예: 같은 모델 점수의 임계값만 다른 것들)이면
                   유효 시행 수는 표면적 전략 개수보다 작다 — 호출부가 근거를
                   대고 정한다.
    trial_sharpes : 그 n_trials회의 시행에서 얻은 (일별, 비연환산) 샤프비
                   추정치 배열. 이로부터 샤프비 추정량의 시행간 분산
                   V[SR_n]을 근사한다(같은 논문의 실무적 근사 — 진짜 독립
                   복제 백테스트가 없을 때 실제 수행한 시행들의 샤프비
                   분산으로 대체). 2개 미만이거나 None이면 분산을 추정할
                   수 없으므로 보정하지 않는다(SR0=0, 다중검정 이전의
                   평범한 PSR과 동일).

    반환: sr_hat(일별 관측 샤프비), sr0(다중검정 기준선), dsr(0~1 확률),
         skew, kurtosis(첨도, 정규분포=3 기준), n_trials, n.
    """
    from scipy import stats as sps

    r = daily_ret.astype(float).dropna().to_numpy()
    n = len(r)
    if n < 2 or r.std(ddof=1) == 0:
        return {"sr_hat": float("nan"), "sr0": float("nan"), "dsr": float("nan"),
                "skew": float("nan"), "kurtosis": float("nan"),
                "n_trials": int(n_trials), "n": int(n)}

    sr_hat = float(r.mean() / r.std(ddof=1))
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))   # 정규분포 = 3

    # 시행간 샤프비 분산 V[SR_n] — trial_sharpes가 2개 이상 있어야 추정 가능
    ts = np.asarray(trial_sharpes, dtype=float) if trial_sharpes is not None else np.array([])
    ts = ts[np.isfinite(ts)]
    var_sr = float(ts.var(ddof=1)) if ts.size >= 2 else 0.0

    if n_trials is not None and n_trials > 1 and var_sr > 0:
        gamma = 0.5772156649015329   # Euler-Mascheroni 상수
        sr0 = float(np.sqrt(var_sr) * (
            (1 - gamma) * sps.norm.ppf(1 - 1.0 / n_trials)
            + gamma * sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
        ))
    else:
        sr0 = 0.0

    denom = np.sqrt(max(1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat ** 2, 1e-12))
    z = (sr_hat - sr0) * np.sqrt(n - 1) / denom
    dsr = float(sps.norm.cdf(z))

    return {"sr_hat": sr_hat, "sr0": sr0, "dsr": dsr,
            "skew": skew, "kurtosis": kurt,
            "n_trials": int(n_trials), "n": int(n)}


def strategy_correlation(daily: pd.DataFrame) -> pd.DataFrame:
    """전략별 일별 수익률의 상관행렬.

    model_top5와 rule_top5_close의 상관이 높으면(대략 0.9 이상) "모델이 단일
    지표가 이미 하는 것 이상을 하지 않는다"는 정성적 주장에 정량적 근거가 된다.
    """
    return daily.corr()


def deflated_sharpe_table(daily: pd.DataFrame, strategy_cols: list[str],
                          n_trials: int) -> pd.DataFrame:
    """strategy_cols 각각에 대해 DSR을 계산해 표로 반환한다.

    시행간 샤프비 분산 V[SR_n] 추정에는 strategy_cols 전체(=실제로 탐색한
    후보군)의 일별(비연환산) 샤프비 분산을 쓴다 — 독립 복제 백테스트가 없을
    때 논문이 제시하는 실무적 근사(실제 수행한 시행들의 관측 샤프비 분산으로
    대체)를 따른다.
    """
    sharpes = []
    for c in strategy_cols:
        r = daily[c].dropna()
        if len(r) > 1 and r.std(ddof=1) > 0:
            sharpes.append(float(r.mean() / r.std(ddof=1)))
    trial_sr = np.array(sharpes)
    rows = []
    for c in strategy_cols:
        out = deflated_sharpe_ratio(daily[c], n_trials=n_trials, trial_sharpes=trial_sr)
        rows.append({"strategy": c, **out})
    return pd.DataFrame(rows)


def market_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """벤치마크: 그날 이벤트 유무와 무관한 전 종목 등가중 익일 시가 수익률.

    FinRL 이 항상 지수(^DJI)와 나란히 보고하는 관행을 따른다 — 전략 수익률만
    단독으로 제시하면 시장이 좋아서 번 것인지 구분할 수 없다.
    """
    from surge_d1 import sources

    ohlcv = sources.load_ohlcv()
    mi = sources.build_market_index(ohlcv)
    gap = mi["gap"].dropna() - config.COST
    out = pd.DataFrame({"ret": gap, "n_picked": mi["n"].reindex(gap.index)})
    return out


def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """전략들을 out-of-sample 예측 위에서 시뮬레이션한다.

    반환: (성과표, 일별수익률표, 비용민감도표)
    """
    oos = out_of_sample_predictions(df)
    mkt = market_benchmark(df)

    # 메인 6종 — 그림 하나에 들어가고, 각각 다른 질문에 답한다
    strategies = {
        "market": mkt,                                     # 시장 하한선
        "all_breakouts": simulate(oos, "score", None),      # ML 없이 룰만
        "rule_top5_close": simulate(oos, "close_strength", 5),   # 단일 지표 (최강 대조군)
        "rule_top5_breakout": simulate(oos, "breakout_pct", 5),  # ML 없는 규칙 선별
        "random_5": simulate_random(oos, 5),                # 우연
        "model_prob50": simulate_threshold(oos, "score", 0.50),  # ML 임계값
        "model_top5": simulate(oos, "score", 5),            # ML 상위N (종목수 맞춤)
    }
    # 부록 — 계산은 하되 메인 그림에는 넣지 않는다
    appendix = {
        "model_prob55": simulate_threshold(oos, "score", 0.55),
        "model_ret_pos": simulate_threshold(oos, "pred_ret", 0.0),
        "model_top10": simulate(oos, "score", 10),
        "rule_top5_value": simulate(oos, "liquidity", 5),
        "model_q80": simulate_threshold(oos, "pick_q80", 1),   # 폴드별 학습구간 상위20%
    }
    strategies.update(appendix)
    bench = mkt["ret"]
    stats = pd.DataFrame(
        [{"strategy": k, **performance_stats(v, benchmark=bench)}
         for k, v in strategies.items()]
    )
    daily = pd.concat(
        [v["ret"].rename(k) for k, v in strategies.items()], axis=1
    ).sort_index()
    sens = cost_sensitivity(oos)
    return stats, daily, sens


# market은 벤치마크이지 "탐색해서 고른 후보"가 아니므로 다중검정 시행에서 뺀다.
# 나머지 11개(메인 6 + 부록 5)가 실제로 우리가 비교·선택을 시도한 후보군이다.
DSR_CANDIDATES = ["all_breakouts", "rule_top5_close", "rule_top5_breakout",
                  "random_5", "model_prob50", "model_top5",
                  "model_prob55", "model_ret_pos", "model_top10",
                  "rule_top5_value", "model_q80"]
# 유효 시행 수: 11개 후보 중 여럿은 같은 신호의 변형이다(같은 prob_cal 점수를
# 임계값·개수만 바꿔 쓰는 model_prob50/model_top5/model_prob55/model_top10/
# model_q80은 사실상 1개 신호). 서로 다른 "신호"는 (1) 모델 점수(prob_cal)
# 계열, (2) 모델 회귀예측(pred_reg, model_ret_pos), (3) close_strength,
# (4) breakout_pct, (5) liquidity, (6) 무선별(all_breakouts) — 6개.
# random_5는 우리가 채택할 수 있는 "후보"가 아니라 순수 우연 대조군이라
# 유효 시행에서도 뺀다.
DSR_N_TRIALS_NAIVE = len(DSR_CANDIDATES)   # 11 — 표면적 전략 개수 그대로
DSR_N_TRIALS_EFFECTIVE = 6                 # 서로 다른 신호 계열 수


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_parquet(config.DATASET_PATH)
    stats, daily, sens = run_backtest(df)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(config.RESULTS_DIR / "backtest_stats.csv",
                 index=False, encoding="utf-8-sig")
    daily.to_csv(config.RESULTS_DIR / "backtest_daily.csv",
                 encoding="utf-8-sig")
    sens.to_csv(config.RESULTS_DIR / "backtest_cost_sensitivity.csv",
                index=False, encoding="utf-8-sig")

    dsr_naive = deflated_sharpe_table(daily, DSR_CANDIDATES, DSR_N_TRIALS_NAIVE)
    dsr_eff = deflated_sharpe_table(daily, DSR_CANDIDATES, DSR_N_TRIALS_EFFECTIVE)
    dsr = dsr_naive.merge(dsr_eff[["strategy", "sr0", "dsr"]], on="strategy",
                          suffixes=("_naive11", "_eff6"))
    dsr.to_csv(config.RESULTS_DIR / "backtest_dsr.csv",
              index=False, encoding="utf-8-sig")

    main_cols = ["market", "all_breakouts", "rule_top5_close",
                "rule_top5_breakout", "random_5", "model_prob50", "model_top5"]
    corr = strategy_correlation(daily[main_cols])
    corr.to_csv(config.RESULTS_DIR / "backtest_correlation.csv",
               encoding="utf-8-sig")

    show = stats.copy()
    for c in ("cum_return", "cagr", "vol", "mdd", "win_rate", "alpha"):
        show[c] = (show[c] * 100).round(2)
    for c in ("sharpe", "sortino", "calmar", "sqn", "beta", "profit_factor"):
        show[c] = show[c].round(3)
    print(show[["strategy", "cum_return", "cagr", "vol", "mdd", "mdd_days",
                "sharpe", "sortino", "calmar", "sqn", "win_rate",
                "profit_factor", "alpha", "beta", "n_days"]].to_string(index=False))
    print("\n=== 비용 민감도 (model_top5) ===")
    print(sens.assign(**{c: (sens[c] * 100).round(2)
                         for c in ("extra_cost", "cum_return", "cagr",
                                   "mdd", "win_rate")}).to_string(index=False))
    print(f"\n=== Deflated Sharpe Ratio (naive n={DSR_N_TRIALS_NAIVE} "
          f"vs effective n={DSR_N_TRIALS_EFFECTIVE}) ===")
    print(dsr[["strategy", "sr_hat", "skew", "kurtosis",
              "sr0_naive11", "dsr_naive11", "sr0_eff6", "dsr_eff6"]]
         .round(4).to_string(index=False))
    print("\n=== 전략 간 일별 수익률 상관행렬 (메인 전략) ===")
    print(corr.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
