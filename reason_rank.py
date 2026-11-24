# -*- coding: utf-8 -*-
"""급등 사유 문장으로 그날 급등주의 익일 시가 수익률 순위를 매긴다.

어닝콜 팩터 구조를 옮긴 것이다:
    사유 문장 → 한국어 금융 언어모델 임베딩 → PCA → Ridge(익일 시가 수익률) → 그날 안에서 순위

    임베딩  kakaobank/kf-deberta-base, 마지막 층 토큰 평균. 고정(미세조정 없음).
            모델 공개(2023-12)가 자료 시작(2024-11)보다 앞서 모델 쪽 미래정보 걱정이 없다.
    타깃    당일 종가 매수 → 익일 시가 매도, 비용 0.21% 차감. 학습 때는 날짜 평균을 빼고
            (그날 안의 상대 순위만 배우게) 학습 구간 1/99%로 자른다.
    검증    확장형 walk-forward, 매월 재학습. PCA와 Ridge 모두 학습 구간 안에서만 적합한다.
    평가    거래일 단위. 일별 순위상관(IC)의 평균과 t, 그리고 그날 상위 1/3 − 하위 1/3 수익 차.

비교 모형 — 임베딩이 값을 더하는지 보려면 텍스트 없는 기준선이 있어야 한다:
    price   당일 등락률 · 상한가 여부 · 거래대금(로그)
    type    재료 유형 더미(키워드 규칙)
    tfidf   글자 n-gram TF-IDF → SVD 32
    emb     kf-deberta → PCA 32                     ← 주 모형(사전에 정함)
    emb+price

구간: 개발 ~2026-08-31, 홀드아웃 2026-09-01~. 홀드아웃은 --holdout 을 줄 때만 계산한다.

실행:
    python scripts/build_reason_panel.py      # 패널 생성
    python reason_rank.py                     # 개발 구간 walk-forward
    python reason_rank.py --holdout           # 홀드아웃 1회 평가
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "data" / "reason_panel.parquet"
EMB = ROOT / "data" / "reason_emb.parquet"
OUT = ROOT / "outputs"

MODEL = "kakaobank/kf-deberta-base"
HOLDOUT_START = pd.Timestamp("2026-09-01")
FIRST_TEST = pd.Timestamp("2025-07-01")     # 그 전 8개월은 첫 학습에만 쓴다
N_COMP = 32
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
MIN_DAY_RET = 0.05     # 정규장 종가 기준 +5% 미만은 '급등주'가 아니다(애프터마켓으로 뽑힌 행 제거)
MIN_NAMES = 5          # 하루 종목이 이보다 적으면 순위상관을 재지 않는다
TYPE_KEYS = {"실적": ("실적", "영업이익", "흑자", "어닝"), "수주": ("수주", "공급계약", "계약 체결", "납품"),
             "지분": ("인수", "합병", "최대주주", "경영권", "지분"), "바이오": ("임상", "FDA", "허가", "기술이전"),
             "정책": ("정부", "정책", "대통령", "법안", "국회"), "자본": ("유상증자", "전환사채", "공개매수", "감자"),
             "테마": ("관련주", "테마", "수혜", "기대감")}


# ---------------------------------------------------------------- 자료
def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PANEL)
    p = p[(p["day_ret"] >= MIN_DAY_RET) & (p["day_ret"] <= 0.32)]
    return p.sort_values(["date", "code"]).reset_index(drop=True)


def embed(texts: list[str]) -> np.ndarray:
    """문장 → 768차원. 한 번 계산한 문장은 data/reason_emb.parquet 에서 다시 읽는다."""
    key = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts]
    cache = pd.read_parquet(EMB).set_index("key") if EMB.exists() else pd.DataFrame()
    todo = sorted({k: t for k, t in zip(key, texts) if k not in cache.index}.items())
    if todo:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL).eval()
        with torch.no_grad():
            for s in range(0, len(todo), 64):
                part = todo[s:s + 64]
                enc = tok([t for _, t in part], padding=True, truncation=True,
                          max_length=128, return_tensors="pt")
                h = model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = ((h * m).sum(1) / m.sum(1)).numpy()               # 패딩을 뺀 토큰 평균
                new = pd.DataFrame(v, index=[k for k, _ in part],
                                   columns=[f"e{i}" for i in range(v.shape[1])])
                cache = pd.concat([cache, new])
                if (s // 64) % 20 == 19 or s + 64 >= len(todo):       # 중간 저장 — 끊겨도 이어서 한다
                    cache.rename_axis("key").reset_index().to_parquet(EMB, index=False)
                    print(f"  임베딩 {min(s + 64, len(todo)):,}/{len(todo):,}", flush=True)
    return cache.loc[key].to_numpy(dtype=np.float32)


def type_dummies(reason: pd.Series) -> np.ndarray:
    return np.column_stack([reason.str.contains("|".join(k)).to_numpy(float)
                            for k in TYPE_KEYS.values()])


def price_feats(p: pd.DataFrame) -> np.ndarray:
    return np.column_stack([p["day_ret"], p["limit_up"].astype(float), np.log(p["tval"])])


# ---------------------------------------------------------------- 학습·예측
def _target(p: pd.DataFrame) -> np.ndarray:
    y = p["y"] - p.groupby("date")["y"].transform("mean")
    lo, hi = y.quantile([0.01, 0.99])
    return y.clip(lo, hi).to_numpy()


def _fit_predict(Xtr, ytr, dtr, Xte, reduce: str | None):
    """reduce: None(그대로) · 'pca' · 'svd'. 차원축소와 표준화는 학습 구간에만 적합한다."""
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if reduce:
        r = (PCA if reduce == "pca" else TruncatedSVD)(n_components=N_COMP, random_state=42).fit(Xtr)
        Xtr, Xte = r.transform(Xtr), r.transform(Xte)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    # 벌점은 학습 구간의 마지막 20% 날짜를 검증으로 떼어 고른다(시간순, 미래 없음)
    days = np.sort(np.unique(dtr))
    cut = days[int(len(days) * 0.8)]
    a, b = dtr < cut, dtr >= cut
    best = max(ALPHAS, key=lambda al: _daily_ic(
        pd.DataFrame({"date": dtr[b], "s": Ridge(alpha=al).fit(Xtr[a], ytr[a]).predict(Xtr[b]),
                      "y": ytr[b]}))[0])
    return Ridge(alpha=best).fit(Xtr, ytr).predict(Xte), best


def _blocks(p: pd.DataFrame, feats: dict[str, np.ndarray], name: str):
    parts = name.split("+")
    X = np.hstack([feats[k] for k in parts])
    reduce = {"emb": "pca", "tfidf": "svd"}.get(parts[0]) if len(parts) == 1 else None
    return X, reduce


def predict_oos(p, feats, name, test_from, test_to, train_mask=None):
    """월별 확장 walk-forward. 각 달의 예측은 그 달 이전 자료로만 만든다."""
    y = _target(p)
    dates = p["date"].to_numpy()
    keep = np.ones(len(p), bool) if train_mask is None else train_mask
    score = np.full(len(p), np.nan)
    months = pd.period_range(test_from, test_to, freq="M")
    for m in months:
        te = (p["date"].dt.to_period("M") == m).to_numpy() & (dates >= np.datetime64(test_from))
        # 라벨이 익일 시가라 학습 마지막 날의 라벨이 검정 첫날과 겹치지 않게 하루 띄운다
        tr = (dates < np.datetime64(m.start_time - pd.Timedelta(days=1))) & keep
        if te.sum() == 0 or tr.sum() < 500:
            continue
        if name == "emb+price":       # 임베딩은 먼저 PCA로 줄인 뒤 가격 변수와 붙인다
            from sklearn.decomposition import PCA
            r = PCA(n_components=N_COMP, random_state=42).fit(feats["emb"][tr])
            X = np.hstack([r.transform(feats["emb"]), feats["price"]])
            score[te], _ = _fit_predict(X[tr], y[tr], dates[tr], X[te], None)
        else:
            X, reduce = _blocks(p, feats, name)
            score[te], _ = _fit_predict(X[tr], y[tr], dates[tr], X[te], reduce)
    return score


# ---------------------------------------------------------------- 평가
def _daily_ic(d: pd.DataFrame) -> tuple[float, float, int]:
    ics = [spearmanr(g["s"], g["y"]).statistic for _, g in d.groupby("date")
           if len(g) >= MIN_NAMES and g["s"].nunique() > 1 and g["y"].nunique() > 1]
    ics = np.array([x for x in ics if np.isfinite(x)])
    if len(ics) < 3:
        return -1.0, 0.0, len(ics)
    return float(ics.mean()), float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), len(ics)


def _spread(d: pd.DataFrame) -> tuple[float, float, float, float]:
    """그날 점수 상위 1/3과 하위 1/3의 평균 수익. 반환: 상위 평균, 하위 평균, 차이 평균, 차이 t."""
    rows = []
    for _, g in d.groupby("date"):
        if len(g) < MIN_NAMES:
            continue
        k = max(1, len(g) // 3)
        g = g.sort_values("s")
        rows.append((g["y"].iloc[-k:].mean(), g["y"].iloc[:k].mean()))
    a = np.array(rows)
    diff = a[:, 0] - a[:, 1]
    return a[:, 0].mean(), a[:, 1].mean(), diff.mean(), diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))


def evaluate(p: pd.DataFrame, score: np.ndarray, mask: np.ndarray) -> dict:
    d = pd.DataFrame({"date": p["date"], "s": score, "y": p["y"]})[mask & np.isfinite(score)]
    ic, t, n = _daily_ic(d)
    top, bot, diff, dt = _spread(d)
    return {"days": n, "rows": len(d), "ic": ic, "ic_t": t, "top": top * 100, "bottom": bot * 100,
            "spread": diff * 100, "spread_t": dt, "base": d["y"].mean() * 100}


def run(p, feats, test_from, test_to, label) -> pd.DataFrame:
    in_test = ((p["date"] >= test_from) & (p["date"] <= test_to)).to_numpy()
    nolim = ~p["limit_up"].to_numpy()
    out = []
    for name in ("price", "type", "tfidf", "emb", "emb+price"):
        s_all = predict_oos(p, feats, name, test_from, test_to)
        out.append({"구간": label, "표본": "전체", "모형": name, **evaluate(p, s_all, in_test)})
        # 상한가는 종가에 살 수 없다 — 학습과 평가 양쪽에서 뺀 결과를 따로 낸다
        s_nl = predict_oos(p, feats, name, test_from, test_to, train_mask=nolim)
        out.append({"구간": label, "표본": "상한가 제외", "모형": name, **evaluate(p, s_nl, in_test & nolim)})
    return pd.DataFrame(out)


def demo() -> None:
    d = pd.DataFrame({"date": np.repeat(pd.date_range("2025-01-01", periods=40), 6),
                      "y": np.tile(np.arange(6.0), 40)})
    d["s"] = d["y"]
    assert abs(_daily_ic(d)[0] - 1.0) < 1e-9          # 점수가 수익 순서와 같으면 IC = 1
    d["s"] = -d["y"]
    assert abs(_daily_ic(d)[0] + 1.0) < 1e-9
    top, bot, diff, _ = _spread(d.assign(s=d["y"]).assign(y=d["y"] + np.random.default_rng(0).normal(0, 1e-6, len(d))))
    assert top > bot and diff > 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="홀드아웃(2026-09-01~) 평가. 한 번만 쓴다")
    args = ap.parse_args()
    demo()

    p = load_panel()
    from sklearn.feature_extraction.text import TfidfVectorizer

    feats = {"emb": embed(p["reason"].tolist()), "price": price_feats(p), "type": type_dummies(p["reason"])}
    # TF-IDF 어휘는 개발 구간 문장으로만 만든다
    dev = (p["date"] < HOLDOUT_START).to_numpy()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True)
    vec.fit(p.loc[dev & (p["date"] < FIRST_TEST), "reason"])
    feats["tfidf"] = vec.transform(p["reason"]).toarray().astype(np.float32)

    last_dev = HOLDOUT_START - pd.Timedelta(days=1)
    print(f"패널 {len(p):,}건 · {p['date'].nunique()}일 · 개발 {dev.sum():,}건 · 홀드아웃 {(~dev).sum():,}건")
    res = run(p, feats, FIRST_TEST, last_dev, "개발")
    if args.holdout:
        res = pd.concat([res, run(p, feats, HOLDOUT_START, p["date"].max(), "홀드아웃")])
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / ("reason_rank_holdout.csv" if args.holdout else "reason_rank_dev.csv"),
               index=False, encoding="utf-8-sig")
