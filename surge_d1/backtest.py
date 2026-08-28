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


def _active(daily: pd.DataFrame) -> pd.Series:
    """실제로 종목을 보유한 날 마스크. n_picked가 없으면 전부 보유로 본다."""
    if "n_picked" not in daily.columns:
        return pd.Series(True, index=daily.index)
    return daily["n_picked"].fillna(0).astype(float) > 0


def _active_win_rate(daily: pd.DataFrame, r: pd.Series) -> float:
    m = _active(daily).reindex(r.index).fillna(False).to_numpy(dtype=bool)
    return float((r[m] > 0).mean()) if m.sum() else float("nan")


def _empty_stats() -> dict:
    keys = ("cum_return cum_return_additive cagr vol sharpe sortino calmar mdd "
            "mdd_days win_rate win_rate_active profit_factor expectancy "
            "mean_trade_ret sqn best_day worst_day alpha beta").split()
    out = {k: float("nan") for k in keys}
    out.update({"cum_return": 0.0, "cum_return_additive": 0.0, "cagr": 0.0,
                "vol": 0.0, "mdd": 0.0, "mdd_days": 0, "n_days": 0,
                "n_trades": 0, "n_active_days": 0})
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
    """일별 수익률 → 성과 지표.

    ⚠️ **읽는 순서에 대한 지침**: `cum_return`/`cagr`는 매일 전액 재투자(복리)를
    가정한다 — 무제한 자본, 매일 5종목 전액 집중, 시장충격 0을 전제하는
    **달성 불가능한 상한**이다(880 거래일·3.6년을 이 가정으로 복리하면 작은
    일평균 우위도 기하급수적으로 부풀려진다). 이 함수가 우선 보고해야 할
    "규모에 무관한(scale-free)" 1차 지표는 `expectancy`(일평균), `sharpe`,
    `mean_trade_ret`(종목당 평균), `mdd`, `win_rate`이고, 복리 계열
    (`cum_return`, `cagr`)은 "상한 참고치"로 격하해 표기한다.
    `cum_return_additive`(복리 없이 일별 수익률을 단순 합산 — 고정 명목원금
    기준)가 복리에 오염되지 않은 누적 지표다.

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
    # cum_return: 매일 전액 재투자(복리) 가정 — 무제한 자본·시장충격 0을
    # 전제하므로 "달성 가능한 상한"으로만 읽어야 한다(보고서 §참고).
    cum_return = float(equity.iloc[-1] - 1.0)
    # 연환산 수익률: (∏(1+r))^(252/n) - 1   (vectorbt annualized_return)
    cagr = float(equity.iloc[-1] ** (TRADING_DAYS / n) - 1.0)
    # cum_return_additive: 복리 없이 일별 수익률을 그냥 더한 값(고정 명목원금
    # 기준). 신호 강도를 복리가 부풀리지 않은 채로 보여주는 1차 지표 —
    # 매일 자본을 100% 재배분한다는 비현실적 가정이 필요 없다.
    cum_return_additive = float(r.sum())

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

    # mean_trade_ret: 포트폴리오-일 단위가 아니라 "개별 종목 매매 1건" 단위
    # 평균 수익률. 하루에 담긴 종목 수(n_picked)가 날마다 다르면 두 평균이
    # 갈릴 수 있다 — 종목 수 가중평균이라 종목이 많이 담긴 날에 더 큰 비중을
    # 준다. n_picked가 없거나(구버전 daily) 전부 0이면 NaN.
    if "n_picked" in daily.columns:
        npk = daily["n_picked"].reindex(r.index).astype(float).fillna(0.0)
        denom = float(npk.sum())
        mean_trade_ret = float((r * npk).sum() / denom) if denom > 0 else float("nan")
    else:
        mean_trade_ret = float("nan")

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
        "cum_return": cum_return, "cum_return_additive": cum_return_additive,
        "cagr": cagr, "vol": vol,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "mdd": mdd, "mdd_days": _max_drawdown_days(equity),
        "win_rate": float((r > 0).mean()),
        # 임계값 전략은 조건 미달일에 현금(수익률 0)으로 남는데, 위 win_rate는
        # 그 현금일을 분모에 넣어 '패배'처럼 센다. 포트폴리오 지표로는 맞지만
        # 매일 만근하는 룰 전략과 나란히 놓으면 오해를 부른다 — 실측으로
        # model_prob50은 880일 중 269일이 현금이라 승률이 크게 희석됐다.
        # 그래서 보유일 기준 승률을 함께 낸다(둘 다 표기하고, 어느 쪽도
        # 대체하지 않는다).
        "n_active_days": int(_active(daily).sum()),
        "win_rate_active": _active_win_rate(daily, r),
        "profit_factor": profit_factor,
        "expectancy": float(r.mean()), "mean_trade_ret": mean_trade_ret,
        "sqn": sqn,
        "best_day": float(r.max()), "worst_day": float(r.min()),
        "alpha": alpha, "beta": beta,
        "n_days": int(n), "n_trades": int(daily["n_picked"].sum()),
    }


def out_of_sample_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """walk-forward 폴드별 검정 구간 예측만 모아 이어 붙인다.

    학습 구간 예측으로 백테스트하면 전부 거짓이므로 반드시 이 경로를 쓴다.

    ⚠️ **재현성(Task 12 재검토)**: LightGBM은 `n_jobs=-1`(다중 스레드)일 때
    히스토그램 분할 순서에 미세한 부동소수점 비결정성이 있고, 이게 880일
    복리를 거치며 눈에 띄게 증폭됐다(재실행마다 CAGR이 수백%p씩 흔들림).
    `surge_d1/models.py`는 다른 작업이 편집 중이라 그 파일의
    `REG_PARAMS`/`CLF_PARAMS`는 건드리지 않고, **이 백테스트 경로 안에서만
    일시적으로 `n_jobs=1`로 덮어썼다가 끝나면 원래 값으로 복구한다** —
    모듈 전역을 저장해뒀다 finally에서 되돌리는 방식이라 예외가 나도 복구된다.
    """
    from surge_d1 import dataset, models, validate

    orig_reg, orig_clf = models.REG_PARAMS, models.CLF_PARAMS
    models.REG_PARAMS = dict(orig_reg, n_jobs=1)
    models.CLF_PARAMS = dict(orig_clf, n_jobs=1)
    try:
        feats = dataset.FEATURE_SETS["M4"]
        parts = []
        for tr_mask, te_mask in validate.walk_forward_folds(df["date"],
                                                            models.TEST_YEARS):
            tr = df[tr_mask].reset_index(drop=True)
            te = df[te_mask].reset_index(drop=True)
            out = models.fit_predict(tr, te, feats)
            te = te.copy()
            # 순위(score)는 **보정 전** 확률로 매긴다. isotonic은 단조라
            # 순위를 보존하지만 계단 함수라 값을 뭉갠다 — 실측으로 4,277행이
            # 고유값 34개로 붕괴했다. 그 상태로 nlargest(5)를 하면 동점
            # 무리 안에서 사실상 무작위 추출이 된다. 절대 확률 판단
            # (임계값·사용자 표시)에만 보정값 prob_cal을 쓴다.
            te["score"] = out["prob_raw"]
            te["prob_cal"] = out["prob_cal"]
            te["pred_ret"] = out["pred_reg"]

            # 분위수 임계값은 **학습 구간 예측**으로 정한다. 검정 구간
            # 예측으로 정하면 결과를 보고 고르는 셈이라 체리피킹이다.
            tr_out = models.fit_predict(tr, tr, feats)
            thr = float(np.quantile(tr_out["prob_raw"], 0.80))
            te["pick_q80"] = (te["score"] >= thr).astype(int)

            parts.append(te)
    finally:
        models.REG_PARAMS, models.CLF_PARAMS = orig_reg, orig_clf

    if not parts:
        return df.head(0).assign(score=[], prob_cal=[], pred_ret=[], pick_q80=[])
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


def picked_liquidity(df: pd.DataFrame, mode: str,
                     score_col: str | None = None,
                     top_n: int | None = None,
                     threshold: float | None = None,
                     n_iter: int = 20, seed: int = config.RANDOM_SEED,
                     value_col: str = "liquidity") -> np.ndarray:
    """전략이 실제로 고른 종목들의 liquidity(5일 평균 거래대금, 원) 값을 모은다.

    자본 배분 여력(capacity)을 가늠하기 위함이다 — 상위 픽이 거래대금이
    얇은 종목에 몰려 있으면, 백테스트가 가정하는 크기의 주문을 실제로는
    체결시킬 수 없다는 뜻이다. simulate()/simulate_threshold()/
    simulate_random()과 정확히 같은 선정 규칙을 재현해 값만 다르게 모은다
    (수익률 대신 liquidity를 모은다는 점만 다르다).

    mode: "topn"(simulate와 동일, top_n=None이면 전부) |
         "threshold"(simulate_threshold와 동일) |
         "random"(simulate_random과 동일, 시드 고정 시 재현 가능)
    """
    vals = []
    if mode == "topn":
        for _, g in df.groupby("date"):
            picked = g if top_n is None else g.nlargest(top_n, score_col)
            if len(picked):
                vals.append(picked[value_col].to_numpy())
    elif mode == "threshold":
        for _, g in df.groupby("date"):
            picked = g[g[score_col] >= threshold]
            if len(picked):
                vals.append(picked[value_col].to_numpy())
    elif mode == "random":
        rng = np.random.default_rng(seed)
        for _ in range(n_iter):
            for _, g in df.groupby("date"):
                k = min(top_n, len(g))
                if k == 0:
                    continue
                idx = rng.choice(len(g), size=k, replace=False)
                vals.append(g[value_col].to_numpy()[idx])
    else:
        raise ValueError(f"알 수 없는 mode: {mode}")
    if not vals:
        return np.array([])
    return np.concatenate(vals)


# 메인 전략들의 실제 선정 규칙 — capacity_table이 picked_liquidity에 그대로
# 넘겨 유동성 분포를 모은다. run_backtest의 strategies 딕셔너리 구성과 반드시
# 짝이 맞아야 한다(선정 규칙이 갈라지면 capacity 수치가 실제 백테스트와
# 어긋난다).
CAPACITY_SPECS = {
    "all_breakouts": dict(mode="topn", score_col="score", top_n=None),
    "rule_top5_close": dict(mode="topn", score_col="close_strength", top_n=5),
    "rule_top5_breakout": dict(mode="topn", score_col="breakout_pct", top_n=5),
    "random_5": dict(mode="random", top_n=5),
    "model_prob50": dict(mode="threshold", score_col="prob_cal", threshold=0.50),
    "model_top5": dict(mode="topn", score_col="score", top_n=5),
}


def capacity_table(oos: pd.DataFrame,
                   specs: dict | None = None) -> pd.DataFrame:
    """메인 전략별 실제 매수 종목의 유동성(거래대금) 분포 요약.

    median만 보면 꼬리의 소형주 쏠림을 놓친다 — p10(하위 10%)을 같이 봐야
    "가장 얇은 픽이 얼마나 얇은지"를 판단할 수 있다.
    """
    specs = CAPACITY_SPECS if specs is None else specs
    rows = []
    for name, kw in specs.items():
        vals = picked_liquidity(oos, **kw)
        if len(vals) == 0:
            rows.append({"strategy": name, "n": 0, "p10": float("nan"),
                        "median": float("nan"), "mean": float("nan")})
            continue
        rows.append({"strategy": name, "n": int(len(vals)),
                    "p10": float(np.quantile(vals, 0.10)),
                    "median": float(np.median(vals)),
                    "mean": float(np.mean(vals))})
    return pd.DataFrame(rows)


def _apply_one_time_cost(ret: pd.Series, cost: float) -> pd.Series:
    """복리 경로 전체를 (1-cost)배로 균일하게 스케일해, 보유기간 전체에
    비용을 딱 한 번만 반영한다.

    첫날 수익률만 (1+r1)(1-cost)-1 로 바꾸면 등가곱 ∏(1+r_t)가 정확히
    원래 등가곱의 (1-cost)배가 된다 — 그 뒤 모든 날은 그대로 둔다.
    매일 비용을 떼는 것과 달리 MDD처럼 '비율'만 보는 지표를 왜곡하지 않는다
    (전체 경로가 상수배로 스케일되면 고점 대비 낙폭 비율은 불변이다).
    """
    out = ret.astype(float).copy()
    if len(out) > 0:
        out.iloc[0] = (1.0 + out.iloc[0]) * (1.0 - cost) - 1.0
    return out


def market_benchmark(df: pd.DataFrame, window=None) -> pd.DataFrame:
    """벤치마크: 시장을 그냥 사서 계속 들고 있었다면(buy & hold).

    FinRL 이 항상 지수(^DJI)와 나란히 보고하는 관행을 따른다 — 전략 수익률만
    단독으로 제시하면 시장이 좋아서 번 것인지 구분할 수 없다.

    ⚠️ **Task 12 재검토로 수정**: 예전 버전은 익일 시가 갭 수익률에서 왕복비용을
    **매일** 뗐다. 이는 계속 들고 있는 벤치마크가 아니라 "매일 사고파는" 전략을
    가정한 것이라 틀렸다 — 실제로 매수보유는 진입·청산 단 두 번만 비용이
    발생한다(그마저도 왕복비용 하나로 이미 표현됨). 그래서:
      1) 계속 들고 있는다는 가정에 맞게 **종가-종가 전 종목 등가중 일별수익률**
         (`build_market_index`의 `ret`, 갭이 아니라 하루 전체 수익률)을 쓴다.
      2) 왕복비용은 보유기간 전체에 **딱 한 번만** 문다 — 복리 경로 전체를
         (1-COST)배로 균일하게 스케일하면 되므로, 첫날 수익률에 한 번만
         반영한다(등가곱 (1+r1)(1+r2)...(1+rT) = (1-COST)×원래 등가곱과
         정확히 같다 — MDD처럼 비율만 보는 지표는 왜곡되지 않는다).

    ⚠️ **최종 검토로 재수정**: 예전 버전은 전 구간(1,879거래일, 2019~2026)을
    썼는데 전략들은 walk-forward 검정 구간(880거래일, 2023~2026)만 덮는다.
    같은 표에 다른 기간의 행을 나란히 두면 사과 대 사과 비교가 아니다 —
    실제로 전 구간 기준 MDD는 40.3%였으나 880일로 맞추면 24.7%다.
    그래서 `window` 로 검정 구간을 받아 **같은 날짜 범위로 잘라낸 뒤** 비용을
    한 번 문다(자르고 나서 비용을 물어야 첫날이 검정 시작일이 된다).
    """
    from surge_d1 import sources

    ohlcv = sources.load_ohlcv()
    mi = sources.build_market_index(ohlcv)
    ret = mi["ret"].dropna()
    if window is not None and len(window):
        lo, hi = pd.Timestamp(min(window)), pd.Timestamp(max(window))
        ret = ret[(ret.index >= lo) & (ret.index <= hi)]
    hold = _apply_one_time_cost(ret, config.COST)
    out = pd.DataFrame({"ret": hold, "n_picked": mi["n"].reindex(hold.index)})
    return out


def market_gap_diagnostic(df: pd.DataFrame, window=None) -> pd.DataFrame:
    """진단용(참고): 시장 전체에 이벤트/모델 전략과 같은 매매방식을 적용했다면.

    이것은 벤치마크가 아니다 — "종가 매수 → 익일 시가 매도"를 시장 전체에
    매일 반복하는, 사실상의 별도 회전매매 전략이다. 그래서 왕복비용도 원래
    방식대로 **매일** 뗀다(model_*/rule_* 전략과 같은 매매방식·같은 비용
    처리를 시장 전체에 적용했을 때의 참고 수치). `market_benchmark`(buy&hold)와
    혼동하지 않도록 별도 함수·별도 이름(`market_gap_daily`)으로 둔다.
    """
    from surge_d1 import sources

    ohlcv = sources.load_ohlcv()
    mi = sources.build_market_index(ohlcv)
    gap = mi["gap"].dropna()
    if window is not None and len(window):
        # 전략들과 같은 검정 구간으로 자른다(최종 검토 Critical 1).
        lo, hi = pd.Timestamp(min(window)), pd.Timestamp(max(window))
        gap = gap[(gap.index >= lo) & (gap.index <= hi)]
    gap = gap - config.COST
    out = pd.DataFrame({"ret": gap, "n_picked": mi["n"].reindex(gap.index)})
    return out


def run_backtest(df: pd.DataFrame
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """전략들을 out-of-sample 예측 위에서 시뮬레이션한다.

    반환: (성과표, 일별수익률표, 비용민감도표, capacity표)

    ⚠️ Task 12 재검토로 capacity표(4번째 원소)가 추가됐다 — 원래는 3-튜플이었다.
    이 함수를 직접 부르는 테스트가 없어(전부 toy 데이터로 단위 함수들만
    테스트) 안전하게 확장했다.
    """
    oos = out_of_sample_predictions(df)
    # 벤치마크를 전략들과 **같은 검정 구간**으로 자른다(최종 검토 Critical 1).
    win = oos["date"]
    mkt = market_benchmark(df, window=win)

    # 메인 6종 — 그림 하나에 들어가고, 각각 다른 질문에 답한다
    strategies = {
        "market": mkt,                                     # 시장 하한선
        "all_breakouts": simulate(oos, "score", None),      # ML 없이 룰만
        "rule_top5_close": simulate(oos, "close_strength", 5),   # 단일 지표 (최강 대조군)
        "rule_top5_breakout": simulate(oos, "breakout_pct", 5),  # ML 없는 규칙 선별
        "random_5": simulate_random(oos, 5),                # 우연
        "model_prob50": simulate_threshold(oos, "prob_cal", 0.50),  # ML 임계값
        "model_top5": simulate(oos, "score", 5),            # ML 상위N (종목수 맞춤)
    }
    # 부록 — 계산은 하되 메인 그림에는 넣지 않는다
    appendix = {
        # 벤치마크가 아니라 진단용 참고치: 시장 전체에 model_*/rule_* 전략과
        # 같은 매매방식(종가매수→익일시가매도, 비용 매일)을 적용했다면.
        "market_gap_daily": market_gap_diagnostic(df, window=win),
        "model_prob55": simulate_threshold(oos, "prob_cal", 0.55),
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
    capacity = capacity_table(oos)
    return stats, daily, sens, capacity


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
    stats, daily, sens, capacity = run_backtest(df)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(config.RESULTS_DIR / "backtest_stats.csv",
                 index=False, encoding="utf-8-sig")
    daily.to_csv(config.RESULTS_DIR / "backtest_daily.csv",
                 encoding="utf-8-sig")
    sens.to_csv(config.RESULTS_DIR / "backtest_cost_sensitivity.csv",
                index=False, encoding="utf-8-sig")
    capacity.to_csv(config.RESULTS_DIR / "backtest_capacity.csv",
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

    # 1차(scale-free) 표 -- 복리 없이 규모에 무관한 지표만. 처음 보이는
    # 숫자가 복리로 부풀려진 값이 되지 않도록 이 표를 가장 먼저 찍는다.
    primary = stats.merge(
        dsr_eff[["strategy", "dsr"]].rename(columns={"dsr": "dsr_eff6"}),
        on="strategy", how="left")
    primary = primary.merge(
        dsr_naive[["strategy", "dsr"]].rename(columns={"dsr": "dsr_naive11"}),
        on="strategy", how="left")
    show1 = primary.copy()
    show1["mean_day_%"] = (show1["expectancy"] * 100).round(3)
    show1["mean_trade_%"] = (show1["mean_trade_ret"] * 100).round(3)
    show1["mdd_%"] = (show1["mdd"] * 100).round(2)
    show1["win_rate_%"] = (show1["win_rate"] * 100).round(2)
    for c in ("sharpe", "dsr_naive11", "dsr_eff6"):
        show1[c] = show1[c].round(4)
    print("=== 1차 지표(규모 무관, 복리 아님) -- 이 결과의 실제 헤드라인 ===")
    print(show1[["strategy", "mean_day_%", "mean_trade_%", "sharpe",
                "dsr_naive11", "dsr_eff6", "mdd_%", "mdd_days", "win_rate_%",
                "n_days", "n_trades"]].to_string(index=False))

    # 2차(복리) 표 -- 참고용 상한치. 매일 전액 재투자·무제한 자본·시장충격
    # 0을 가정하므로 실제로 달성 가능한 수치가 아니다. 1차 표 다음에만,
    # "상한 참고치"라는 라벨과 함께 보여준다.
    show2 = stats.copy()
    show2["cum_return_additive_%"] = (show2["cum_return_additive"] * 100).round(2)
    show2["cagr_%(상한,비현실적)"] = (show2["cagr"] * 100).round(1)
    show2["cum_return_%(상한,비현실적)"] = (show2["cum_return"] * 100).round(1)
    print("\n=== 2차 지표(복리) -- 경고: 매일 전액 재투자·무제한 자본·"
          "시장충격 0을 가정한 '달성 불가능한 상한'일 뿐, 실현 가능한 "
          "수익 전망이 아니다. cum_return_additive_%(단순 합산, 고정 "
          "명목원금 기준)만이 복리에 오염되지 않은 누적 지표다 ===")
    print(show2[["strategy", "cum_return_additive_%",
                "cagr_%(상한,비현실적)",
                "cum_return_%(상한,비현실적)"]].to_string(index=False))

    print("\n=== Capacity(자본 여력) -- 실제로 고른 종목의 5일평균 거래대금"
          "(liquidity, 억원) ===")
    show_cap = capacity.copy()
    for c in ("p10", "median", "mean"):
        show_cap[c] = (show_cap[c] / 1e8).round(1)   # 원 -> 억원
    show_cap = show_cap.rename(columns={"p10": "p10_억원", "median": "median_억원",
                                        "mean": "mean_억원"})
    print(show_cap.to_string(index=False))

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
