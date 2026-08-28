# -*- coding: utf-8 -*-
"""이벤트 × 피처 × 라벨을 조립해 parquet으로 저장한다. 스펙 §8.

F4(재료)는 폴드별 point-in-time으로 생성해야 하므로 여기서는 계산하지 않는다.
원본 컬럼(ctype, reason, d1, has_catalyst)만 붙여두고 Task 11이 폴드마다
CatalystIndex를 학습 데이터로 fit해 F4를 만든다. 전체 기간에 대해 F4를
한 번만 계산해 두면 검정 구간의 재료가 학습 구간의 prior/analog에 새어
들어가는 미래정보 누수이므로, 이 파이프라인은 F4를 절대 계산하지 않는다.
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from surge_d1 import (config, events, features_f1, features_f2,
                      features_f3, features_f4, features_f5, sources)

FEATURE_SETS = {
    "M1": list(features_f1.F1_NAMES),
    "M2": list(features_f1.F1_NAMES) + list(features_f2.F2_NAMES),
    "M3": (list(features_f1.F1_NAMES) + list(features_f2.F2_NAMES)
           + list(features_f3.F3_NAMES)),
}
FEATURE_SETS["M4"] = FEATURE_SETS["M3"] + list(features_f5.F5_NAMES)   # +수급
FEATURE_SETS["M5"] = FEATURE_SETS["M4"] + list(features_f4.F4_NAMES)   # +재료

# parquet에 미리 계산해 둘 피처. F4(재료)는 폴드별 point-in-time 생성이라 제외한다.
BASE_FEATURES = FEATURE_SETS["M4"]

# 절대 피처가 되면 안 되는 컬럼 (라벨과 그 파생, 식별자). dataset이 이 검사를
# 직접 수행한다 — d1은 재료 CSV의 익일 수익률(=라벨과 사실상 동의어)이라
# Task 11이 CatalystIndex를 fit할 때는 쓰지만 피처 목록에는 절대 들어가면
# 안 된다.
BANNED_AS_FEATURE = {"d1", "y_raw", "y_hold", "y_hold20", "y_mkt", "y_cls",
                     "y_up", "close", "i", "date", "code"}


def add_sample_weight(df: pd.DataFrame) -> pd.DataFrame:
    """같은 날 이벤트들이 시장 충격을 공유하므로 1/(그날 건수)로 가중. 스펙 §7.3."""
    out = df.copy()
    cnt = out.groupby("date")["date"].transform("size")
    out["weight"] = 1.0 / cnt
    return out


def _check_no_label_leak() -> None:
    """라벨 누수 방어선: 피처 목록에 라벨이 섞이면 결과가 전부 무효다.

    build_dataset()의 맨 앞에서 검사한다 — 이 가드가 실패해야 할 때
    23,654건에 대한 피처 계산(수 분)을 기다릴 이유가 없다.
    """
    for name, feats_list in FEATURE_SETS.items():
        leak = set(feats_list) & BANNED_AS_FEATURE
        if leak:
            raise RuntimeError(f"{name} 피처셋에 라벨이 섞였다: {leak}")


def build_dataset() -> pd.DataFrame:
    """전체 파이프라인: 이벤트 → 피처 → 라벨 → 재료 조인 → 가중."""
    _check_no_label_leak()

    ohlcv = sources.load_ohlcv()
    market = sources.build_market_index(ohlcv)
    inv = features_f5.load_investor()
    ev = events.collect_events(ohlcv, exclude=events.load_excluded_codes())
    ev = events.add_market_adjusted(ev)

    rows = []
    t0 = time.time()
    n_total = len(ev)
    for n, r in enumerate(ev.itertuples(index=False), 1):
        df = ohlcv[r.code]
        i = int(r.i)
        feat = {}
        feat.update(features_f1.compute_f1(df, i, market))
        feat.update(features_f2.compute_f2(df, i, market))
        feat.update(features_f3.compute_f3(df, i))
        feat.update(features_f5.compute_f5(inv.get(r.code), r.date))
        rows.append(feat)
        if n % 5000 == 0 or n == n_total:
            print(f"  피처 {n}/{n_total} · {time.time()-t0:.0f}s", flush=True)

    feats = pd.DataFrame(rows, index=ev.index)
    out = pd.concat([ev, feats], axis=1)

    cat = sources.load_catalyst()
    out = features_f4.attach_catalyst(out, cat)
    out = add_sample_weight(out)

    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    df = build_dataset()
    config.DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.DATASET_PATH)
    print(f"저장: {config.DATASET_PATH}")
    print(f"행 {len(df)} · 열 {len(df.columns)}")
    print(f"재료 매칭 {int(df['has_catalyst'].sum())}건 "
          f"({df['has_catalyst'].mean()*100:.1f}%)")
    miss = df[BASE_FEATURES].isna().mean()
    print(f"최대 결측률 {miss.max()*100:.2f}% ({miss.idxmax()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
