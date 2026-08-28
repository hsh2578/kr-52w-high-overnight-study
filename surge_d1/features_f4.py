# -*- coding: utf-8 -*-
"""F4 — 재료 피처 5개. 스펙 §5.4.

핵심 제약: type_prior와 TF-IDF 인덱스는 **학습 구간 데이터만으로** 만든다.
전역 상수(CATALYST_TYPE_PRIOR)를 쓰면 미래정보 누수이므로 쓰지 않는다.
원 아이디어: 모의투자대회/catalyst_engine/baserate.py (char n-gram TF-IDF).

CatalystIndex.fit()에 넘긴 rows 바깥의 어떤 데이터도 transform() 결과에
영향을 주지 않는다 — fit이 곧 폴드 경계다. 호출자는 학습 폴드 재료만
fit에 넘겨야 하며, 검정/전체 폴드 재료를 fit에 섞으면 미래정보 누수다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

F4_NAMES = ["type_prior", "analog_mean_d1", "analog_win_rate",
            "analog_n", "reason_len"]

NEUTRAL = {"type_prior": 0.0, "analog_mean_d1": 0.0,
           "analog_win_rate": 0.5, "analog_n": 0.0, "reason_len": 0.0}


class CatalystIndex:
    """학습 구간 재료 데이터로만 만든 prior + 유사사례 검색기."""

    def __init__(self) -> None:
        self._fitted = False
        self._type_mean: dict[str, float] = {}
        self._global_mean = 0.0
        self._vec = None
        self._mat = None
        self._d1 = np.array([])

    def fit(self, rows: pd.DataFrame) -> "CatalystIndex":
        """rows: 컬럼 ctype, reason, d1 (학습 구간만 넘길 것)."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        r = rows.dropna(subset=["d1"]).copy()
        r["reason"] = r["reason"].fillna("").astype(str)
        r = r[r["reason"].str.len() > 0]

        self._global_mean = float(r["d1"].mean()) if len(r) else 0.0
        self._type_mean = r.groupby("ctype")["d1"].mean().to_dict()
        self._d1 = r["d1"].to_numpy(dtype=float)

        if len(r) >= 2:
            self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                        min_df=1, sublinear_tf=True)
            self._mat = self._vec.fit_transform(r["reason"].tolist())
        self._fitted = True
        return self

    def transform(self, ctype, reason, k: int = 10) -> dict:
        """재료 유형·사유 → F4 피처 5개. 재료가 없으면 중립값."""
        if not self._fitted:
            raise RuntimeError("CatalystIndex.fit()을 먼저 호출해야 한다")
        if not reason or (isinstance(reason, float) and np.isnan(reason)):
            return dict(NEUTRAL)

        reason = str(reason)
        out = dict(NEUTRAL)
        out["type_prior"] = float(self._type_mean.get(ctype, self._global_mean))
        out["reason_len"] = float(len(reason))

        if self._mat is not None:
            from sklearn.metrics.pairwise import cosine_similarity

            sims = cosine_similarity(self._vec.transform([reason]), self._mat)[0]
            kk = int(min(k, len(sims)))
            idx = np.argsort(-sims)[:kk]
            vals = self._d1[idx]
            if vals.size:
                out["analog_mean_d1"] = float(vals.mean())
                out["analog_win_rate"] = float((vals > 0).mean())
                out["analog_n"] = float(vals.size)
        return out


def attach_catalyst(events: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """이벤트에 (date, code) 기준으로 재료를 left join한다."""
    c = cat[["date", "code", "ctype", "reason", "d1"]].drop_duplicates(
        subset=["date", "code"], keep="first")
    out = events.merge(c, on=["date", "code"], how="left")
    out["has_catalyst"] = out["reason"].notna().astype(int)
    return out
