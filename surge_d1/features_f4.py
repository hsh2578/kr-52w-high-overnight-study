# -*- coding: utf-8 -*-
"""F4 — 재료 피처 5개. 스펙 §5.4.

핵심 제약 1(폴드 경계): type_prior와 TF-IDF 인덱스는 **학습 구간
데이터만으로** 만든다. 전역 상수(CATALYST_TYPE_PRIOR)를 쓰면 미래정보
누수이므로 쓰지 않는다. 원 아이디어: 모의투자대회/catalyst_engine/baserate.py
(char n-gram TF-IDF).

CatalystIndex.fit()에 넘긴 rows 바깥의 어떤 데이터도 transform() 결과에
영향을 주지 않는다 — fit이 곧 폴드 경계다. 호출자는 학습 폴드 재료만
fit에 넘겨야 하며, 검정/전체 폴드 재료를 fit에 섞으면 미래정보 누수다.

핵심 제약 2(자기포함 누수): Task 11은 fit(학습 폴드)한 인덱스로 **학습
폴드 자신의 행에 대한 피처도** 만들어야 한다(모델이 학습 피처가 필요하므로).
그런데 학습 폴드의 재료가 있는 행은 fit()이 본 후보 풀에 자기 자신도
들어있으므로, 아무 조치가 없으면 코사인 유사도 1.0으로 반드시 자기 자신이
최근접 이웃이 되어 자기 자신의 d1이 analog_mean_d1/analog_win_rate와
type_prior(그 유형 평균)에 그대로 새어 들어온다 — 통제되지 않은 타겟
인코딩이다. 이를 막으려면 `transform(..., exclude_key=...)`로 그 행
자신의 키(=fit()에 넘긴 rows의 인덱스 라벨)를 넘겨 자기 자신을 후보
풀과 유형 평균 양쪽에서 제외해야 한다.

Task 11 호출 규약: fit()에 넘긴 학습 폴드 DataFrame의 인덱스를 그대로
유지한 채(reset_index 하지 말고), 학습·검정 양쪽 폴드 전체에 대해 피처를
만들 때 매 행마다 `exclude_key=row_index`(그 행 자신의 인덱스 라벨)를
넘기면 된다. 검정 폴드 행의 인덱스는 애초에 fit() 집합에 없으므로
exclude_key를 찾지 못해 아무 것도 제외되지 않고 결과가 기존과 완전히
동일하다 — 즉 학습/검정을 구분하는 조건문 없이 항상 exclude_key=row_index를
넘기는 한 가지 호출 패턴으로 양쪽을 다 올바르게 처리할 수 있다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

F4_NAMES = ["type_prior", "analog_mean_d1", "analog_win_rate",
            "analog_n", "reason_len"]

NEUTRAL = {"type_prior": 0.0, "analog_mean_d1": 0.0,
           "analog_win_rate": 0.5, "analog_n": 0.0, "reason_len": 0.0}


class CatalystIndex:
    """학습 구간 재료 데이터로만 만든 prior + 유사사례 검색기.

    transform()의 exclude_key로 질의 행 자신을 후보 풀·유형 평균에서
    제외할 수 있다 — 모듈 docstring의 "핵심 제약 2" 참조.
    """

    def __init__(self) -> None:
        self._fitted = False
        self._keys = np.array([])
        self._key_to_pos: dict = {}
        self._ctypes = np.array([])
        self._d1 = np.array([])
        self._type_sum: dict[str, float] = {}
        self._type_count: dict[str, int] = {}
        self._global_sum = 0.0
        self._global_count = 0
        self._vec = None
        self._mat = None

    def fit(self, rows: pd.DataFrame) -> "CatalystIndex":
        """rows: 컬럼 ctype, reason, d1 (학습 구간만 넘길 것).

        rows.index를 그대로 "키"로 보존한다 — transform(exclude_key=...)가
        이 키로 자기 자신의 행을 찾아 후보 풀에서 제외한다. 인덱스가
        중복되면 첫 항목만 키로 등록된다(자기 배제 목적으로는 충분하다).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        r = rows.dropna(subset=["d1"]).copy()
        r["reason"] = r["reason"].fillna("").astype(str)
        r = r[r["reason"].str.len() > 0]

        self._keys = r.index.to_numpy()
        self._ctypes = r["ctype"].to_numpy()
        self._d1 = r["d1"].to_numpy(dtype=float)

        self._key_to_pos = {}
        for i, key in enumerate(self._keys):
            if key not in self._key_to_pos:
                self._key_to_pos[key] = i

        self._global_sum = float(self._d1.sum())
        self._global_count = int(self._d1.size)
        self._type_sum = r.groupby("ctype")["d1"].sum().to_dict()
        self._type_count = r.groupby("ctype")["d1"].count().to_dict()

        if len(r) >= 2:
            self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                        min_df=1, sublinear_tf=True)
            self._mat = self._vec.fit_transform(r["reason"].tolist())
        self._fitted = True
        return self

    def transform(self, ctype, reason, k: int = 10, exclude_key=None) -> dict:
        """재료 유형·사유 → F4 피처 5개. 재료가 없으면 중립값.

        exclude_key: fit()에 넘긴 rows의 인덱스 라벨. 지정하면 그 키에
        해당하는 행을 후보 풀(analog_*)과 유형 평균(type_prior) 양쪽에서
        제외한다 — 질의 행 자신이 fit 집합 안에 있을 때(학습 폴드 피처를
        만들 때) 자기 자신의 라벨이 새어 들어오는 것을 막기 위함이다.
        fit 집합에 없는 키를 넘기면(검정 폴드의 정상 경로) 아무 효과가
        없어 exclude_key를 안 넘긴 것과 결과가 완전히 같다.
        """
        if not self._fitted:
            raise RuntimeError("CatalystIndex.fit()을 먼저 호출해야 한다")
        if not reason or (isinstance(reason, float) and np.isnan(reason)):
            return dict(NEUTRAL)

        reason = str(reason)
        out = dict(NEUTRAL)
        out["reason_len"] = float(len(reason))

        excl_pos = None
        excl_d1 = None
        excl_ctype = None
        if exclude_key is not None and exclude_key in self._key_to_pos:
            excl_pos = self._key_to_pos[exclude_key]
            excl_d1 = float(self._d1[excl_pos])
            excl_ctype = self._ctypes[excl_pos]

        # --- type_prior: 유형 평균에서 자기 자신을 뺀 뒤 평균을 낸다.
        # 그 유형의 표본이 하나도 안 남으면(=자기 자신이 유일한 멤버였으면)
        # 전역 평균으로 대체하되, 그 전역 평균에서도 자기 자신을 뺀다
        # (그렇지 않으면 대체값을 통해 자기 라벨이 다시 새어 들어온다).
        t_count = self._type_count.get(ctype, 0)
        t_sum = self._type_sum.get(ctype, 0.0)
        g_count = self._global_count
        g_sum = self._global_sum
        if excl_pos is not None:
            g_count -= 1
            g_sum -= excl_d1
            if excl_ctype == ctype:
                t_count -= 1
                t_sum -= excl_d1
        if t_count > 0:
            out["type_prior"] = t_sum / t_count
        elif g_count > 0:
            out["type_prior"] = g_sum / g_count
        else:
            out["type_prior"] = 0.0

        # --- analog_*: TF-IDF 코사인 유사도 최근접 k개. 제외 대상은 유사도를
        # -inf로 눌러 후보 풀에서 빼고, 풀 크기도 그만큼 줄여서 k와 비교한다
        # (그래야 analog_n이 "요청한 k"가 아니라 "실제로 쓴 이웃 수"가 된다).
        if self._mat is not None:
            from sklearn.metrics.pairwise import cosine_similarity

            sims = cosine_similarity(self._vec.transform([reason]), self._mat)[0]
            pool = len(sims)
            if excl_pos is not None:
                sims = sims.copy()
                sims[excl_pos] = -np.inf
                pool -= 1
            kk = int(min(k, pool))
            if kk > 0:
                idx = np.argsort(-sims)[:kk]
                vals = self._d1[idx]
                out["analog_mean_d1"] = float(vals.mean())
                out["analog_win_rate"] = float((vals > 0).mean())
                out["analog_n"] = float(vals.size)
        return out


def attach_catalyst(events: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """이벤트에 (date, code) 기준으로 재료를 left join한다.

    tie-break 메모: drop_duplicates(keep="first")는 같은 (date, code)에
    재료가 두 건 이상 있을 때 CSV상 먼저 나오는 행을 남긴다. 이 순서는
    CSV 원본의 등장 순서일 뿐 의미 있는 기준(예: 중요도)이 아니다 —
    실측 데이터에서는 (date, code) 중복이 없어 이 분기 자체가 발동하지
    않으므로 동작을 바꾸지 않는다.
    """
    c = cat[["date", "code", "ctype", "reason", "d1"]].drop_duplicates(
        subset=["date", "code"], keep="first")
    out = events.merge(c, on=["date", "code"], how="left")
    out["has_catalyst"] = out["reason"].notna().astype(int)
    return out
