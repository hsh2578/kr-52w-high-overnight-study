# -*- coding: utf-8 -*-
"""오늘 52주 신고가를 돌파한 종목의 익일 순이익 확률·기대수익률을 산출한다.

학습은 저장된 데이터셋 전체(단, 기준일 이전만)로 하고, 예측은 최신 봉 후보에
대해 한다. 익일 데이터가 없으므로 라벨 없이 피처만 계산한다.

주의(핵심 함정): events.is_breakout은 i+1<len(df)를 요구해 데이터의 마지막
봉에는 쓸 수 없다(다음날 라벨을 계산해야 하므로). 오늘자 예측은 다음날이
존재하지 않는 상태에서 돌파를 판정해야 하므로, 이 파일은 "익일 존재" 조건만
뺀 별도 판정(_is_breakout_last)을 쓴다. 그 외 조건(신고가·거래량 1.5배·
거래대금 100억·i>=260·최근 HALT_LOOKBACK봉 내 거래정지 없음(오늘 포함)·
가격제한폭 이내·비상장주(ETF 등) 제외)은 연구 스크리닝(events.is_breakout,
events.load_excluded_codes)과 정확히 동일해야 한다 — 그렇지 않으면 라이브
데모가 논문의 결과와 다른 모집단을 스크리닝하게 된다.

출력 지표 정의(스펙 §6, §7.4 참조):
    익일 순이익 확률 = prob_cal (보정된 확률). "익일 상승 확률"이 아니다 —
    y_cls는 왕복비용(COST=0.21%)을 차감한 뒤 양수인지로 정의되므로, 이
    확률은 "오르는가"가 아니라 "비용을 제하고도 남는가"를 답한다. 연구
    결과상 익일 상승확률(gross)은 47.6%, 이 확률(net)은 44.6%로 서로
    다르다 — 혼동해 보고하면 안 된다.
    기대수익률 = pred_reg (비용 차감 후 기대 순수익률, y_raw 회귀 예측).

사용:
    python predict.py                  # 캐시 최신일 기준
    python predict.py --asof 20260828  # 특정일 기준
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from surge_d1 import (config, dataset, events, features_f1, features_f2,
                      features_f3, features_f5, models, sources)

CAUTION = ("주의: 이 수치는 연구용 추정치(검정 폴드 AUC 약 0.60, "
          "학습-검정 성능 격차 약 0.22)이며 투자 권유가 아니다.")


def _is_breakout_last(df: pd.DataFrame, i: int) -> bool:
    """익일 존재 조건만 뺀 돌파 판정 (예측 전용).

    events.is_breakout과 동일한 조건을 그대로 검사하되:
      - i+1 < len(df) 요구를 뺀다 (오늘이 마지막 봉이어도 판정 가능)
      - 거래정지 lookback 창에서 "익일" 봉을 보지 않는다 — 오늘까지만 본다
        (원래는 i-4..i+1 구간, 여기서는 i-4..i 구간).
    나머지(신고가·거래량 1.5배·거래대금 100억·i>=260·가격제한폭)는
    events.is_breakout과 정확히 동일한 조건·상수를 쓴다.
    """
    if i < config.MIN_BAR_INDEX or i >= len(df):
        return False

    if "halted" in df.columns:
        lo = max(0, i - config.HALT_LOOKBACK + 1)
        if bool(df["halted"].iloc[lo:i + 1].any()):
            return False

    c = float(df["Close"].iloc[i])
    v = float(df["Volume"].iloc[i])
    if not np.isfinite(c) or not np.isfinite(v) or c <= 0:
        return False

    prev_c = float(df["Close"].iloc[i - 1])
    if np.isfinite(prev_c) and prev_c > 0:
        if abs(c / prev_c - 1.0) > config.MAX_DAILY_MOVE:
            return False

    prior_high = float(df["Close"].iloc[i - config.HIGH_52W_PERIOD:i].max())
    if not (c > prior_high):
        return False

    vol_avg = float(np.nanmean(df["Volume"].iloc[i - 50:i].to_numpy(dtype=float)))
    if not (np.isfinite(vol_avg) and vol_avg > 0
            and v >= config.VOLUME_SURGE_MULT * vol_avg):
        return False

    return bool(c * v >= config.MIN_TRADING_VALUE)


def find_today_events(ohlcv: dict, market: pd.DataFrame,
                      asof: pd.Timestamp) -> pd.DataFrame:
    """asof 날짜에 돌파한 종목과 그 피처(M4)를 모은다. 라벨은 계산하지 않는다."""
    exclude = events.load_excluded_codes()
    inv = features_f5.load_investor()

    rows = []
    for code, df in ohlcv.items():
        if code in exclude:
            continue
        if asof not in df.index:
            continue
        i = int(df.index.get_loc(asof))
        if not _is_breakout_last(df, i):
            continue
        feat = {}
        feat.update(features_f1.compute_f1(df, i, market))
        feat.update(features_f2.compute_f2(df, i, market))
        feat.update(features_f3.compute_f3(df, i))
        feat.update(features_f5.compute_f5(inv.get(code), asof))
        rows.append({"code": code, "date": asof,
                     "close": float(df["Close"].iloc[i]), **feat})
    return pd.DataFrame(rows)


def _assert_train_before_asof(train: pd.DataFrame, asof: pd.Timestamp) -> None:
    """학습 데이터에 기준일 이후 행이 섞이지 않았는지 명시적으로 검증한다.

    "date < asof 필터를 썼으니 당연히 맞겠지"에 기대지 않고, 실제로
    max(date) < asof 인지 어서션으로 확인한다 — 필터 로직 자체의 버그를
    잡기 위한 것이다.
    """
    if len(train) == 0:
        return
    max_date = pd.to_datetime(train["date"]).max()
    assert max_date < asof, (
        f"학습 데이터 최대일({max_date.date()})이 기준일({asof.date()}) "
        "이후 — 미래정보 누수.")


def load_training(asof: pd.Timestamp) -> pd.DataFrame:
    """기준일 이전(strictly before)의 학습 데이터만 읽는다."""
    df = pd.read_parquet(config.DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"] < asof].reset_index(drop=True)
    _assert_train_before_asof(train, asof)
    return train


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="YYYYMMDD (기본: 캐시 최신일)")
    args = ap.parse_args()

    ohlcv = sources.load_ohlcv()
    market = sources.build_market_index(ohlcv)
    asof = (pd.Timestamp(args.asof) if args.asof
           else max(df.index[-1] for df in ohlcv.values()))
    print(f"기준일: {asof.date()}")

    cand = find_today_events(ohlcv, market, asof)
    if cand.empty:
        print("해당일 돌파 종목 없음")
        return 0
    print(f"돌파 종목 {len(cand)}개")

    train = load_training(asof)
    print(f"학습 표본 {len(train)}건 (최대일: "
         f"{pd.to_datetime(train['date']).max().date() if len(train) else '없음'})")
    feats = dataset.FEATURE_SETS["M4"]
    out = models.fit_predict(train, cand, feats)

    cand = cand.copy()
    cand["prob_cal"] = out["prob_cal"]
    cand["pred_reg"] = out["pred_reg"]
    res = (cand[["code", "date", "close", "prob_cal", "pred_reg"]]
          .rename(columns={"prob_cal": "익일 순이익 확률",
                           "pred_reg": "기대수익률"})
          .sort_values("익일 순이익 확률", ascending=False)
          .reset_index(drop=True))
    res["익일 순이익 확률"] = (res["익일 순이익 확률"] * 100).round(1)
    res["기대수익률"] = (res["기대수익률"] * 100).round(3)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"predict_{asof.strftime('%Y%m%d')}.csv"
    res.to_csv(path, index=False, encoding="utf-8-sig")
    print(res.head(20).to_string(index=False))
    print(f"\n저장: {path}")
    print(CAUTION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
