# -*- coding: utf-8 -*-
"""F3 — Microsoft qlib Alpha158 계열 피처 20개. 스펙 §5.3.

원 정의: https://github.com/microsoft/qlib — qlib/contrib/data/loader.py 의
Alpha158DL.get_feature_config(). KBAR 9종은 원식 그대로, 롤링은 d=20 창의
11종을 옮겼다(스펙은 d=5,20 두 창을 언급하지만 20개 총원에 맞추려고
**롤링은 d=20 하나만** 쓴다 — KBAR 9 + 롤링 11 = 20). 0으로 나누는 경우
qlib과 동일하게 분모에 1e-12(EPS)를 더한다. 재현이 목적이므로 정의를
"개선"하지 않는다 — 이상해 보이는 식도 그대로 옮기고 리포트에 기록한다.

NaN(거래정지 봉) 처리 방침 — 이 모듈에서 가장 중요한 결정
----------------------------------------------------------
sources.mark_halted 규칙: 거래정지 봉은 Open/High/Low/Volume이 NaN이고
Close만 직전값을 유지한다. 20봉 롤링 창에 정지봉이 섞이는 것은 일상적이다
(전체의 약 1.6%). 브리프에 적힌 구현을 그대로 쓰면 np.corrcoef/np.polyfit/
np.quantile/np.argmax/.std()/.mean() 이 전부 NaN을 전파하거나 오동작하므로,
그대로 옮기지 않고 다음 원칙으로 바꿔 구현한다:

1. events.is_breakout()은 이벤트 당일(i)을 포함해 i-4..i+1 구간에 정지봉이
   있으면 이벤트로 인정하지 않는다(HALT_LOOKBACK=5). 따라서 **봉 i 자신은
   항상 유효(비정지)** 하다 — compute_f3에 들어오는 o,h,l,c는 항상 유한하고,
   20봉 창(i-19..i)도 최소 1개(=i 자신)는 항상 유효한 원소를 갖는다. 즉
   "창 전체가 NaN"인 경우는 이 데이터 구조상 발생하지 않는다.
2. Close는 정지봉에서도 NaN이 아니므로(직전값 유지), Close만 쓰는 계산
   (CNTP_20, SUMP_20, BETA_20, RSQR_20, QTLU_20과 그 바탕이 되는 dc)은
   원래도 NaN에 노출되지 않는다. 그래도 방어적으로 nan-safe 버전을 쓴다.
3. High/Low/Volume을 쓰는 계산(RSV_20, CORR_20/CORD_20의 거래량 쪽,
   WVMA_20, VSTD_20, IMAX_20)은 np.nanmin/np.nanmax/np.nanstd/np.nanmean/
   np.nanargmax로 바꾸거나, 상관·회귀처럼 두 배열을 함께 쓰는 경우는
   두 배열 모두 유한한 위치만 골라 마스킹한 뒤 계산한다. 이렇게 하면
   부분적으로 정지된 창도 "유효한 나머지 봉들만으로" 값을 내고, 결측이나
   엉뚱한 상수로 무너지지 않는다.
4. RSV_20은 과거 버그(정지봉 Low=0)가 정확히 이 피처를 오염시켰던 지점이다
   (지금은 Low가 0이 아니라 NaN이라 원인은 다르지만, 창 안의 결측을
   min/max에서 제대로 걸러내지 않으면 같은 방식으로 재발한다). nanmin/
   nanmax로 막는다.

_safe()가 0.0으로 접는 이유
----------------------------
`_safe(x)`는 최종적으로 비유한값을 0.0으로 접는다. F2(features_f2.py)는
반대로 "진짜 결측"을 0.0으로 메우지 않고 NaN을 그대로 전달했다(0%는
"관측된 평균 갭이 0%"라는 별개의 주장이 되어버리기 때문). F3에서 0.0으로
접는 선택이 그 원칙과 모순되지 않는 이유:

- 원칙 1에서 보였듯 이 데이터 구조에서는 20봉 창이 통째로 결측이 되는
  경우가 없다(이벤트 봉 i가 항상 유효). 따라서 "정말 계산할 수 없어서
  NaN"인 경우가 거의 없다 — 있다면 그건 데이터 결측이 아니라 수학적으로
  정의되지 않는 경우(예: 표준편차 0인 두 벡터의 상관계수, 분산이 0인
  회귀)뿐이다.
- 그런 수학적 미정의 케이스는 이미 각 계산 안에서 명시적으로 0.0으로
  분기 처리한다(qlib 원 구현과 동일: "상관 없음/기울기 없음"을 0으로
  둔다). 즉 _safe의 0.0 폴백은 "결측을 지운다"가 아니라 "부동소수점
  연산이 뜻밖의 inf/NaN을 냈을 때의 최후 방어선"이라는 역할만 한다 —
  정상 경로에서는 사실상 도달하지 않는다.
- LightGBM은 NaN을 네이티브로 다루지만(결측 자체를 분기 정보로 학습),
  여기서는 결측 자체가 신호가 될 상황(원칙 1)이 없으므로 NaN을 굳이
  보존할 이유가 없다. 반대로 KBAR처럼 항상 유한해야 하는 값이 실수로
  inf가 되는 것을 조용히 0으로 눌러버리면 디버깅이 어려워질 위험은
  있지만, 이는 테스트(전 창 NaN 섞기)로 방어한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
WINDOW = 20

KBAR_NAMES = ["KMID", "KLEN", "KMID2", "KUP", "KUP2",
              "KLOW", "KLOW2", "KSFT", "KSFT2"]
ROLL_NAMES = [f"{n}_{WINDOW}" for n in
              ["RSV", "CORR", "CORD", "CNTP", "SUMP",
               "WVMA", "VSTD", "BETA", "RSQR", "IMAX", "QTLU"]]
F3_NAMES = KBAR_NAMES + ROLL_NAMES


def _kbar(o: float, h: float, l: float, c: float) -> dict:
    """qlib Alpha158 KBAR 9종.

    o(시가)는 분모에 쓰인다. events.is_breakout이 이벤트 당일을 정지봉에서
    배제하므로 여기 들어오는 o는 항상 유한하고 0보다 크다 — 그래도 qlib과
    동일하게 EPS를 더해 이론적 0-나눗셈을 막는다.
    """
    hl = (h - l) + EPS
    return {
        "KMID": (c - o) / (o + EPS),
        "KLEN": (h - l) / (o + EPS),
        "KMID2": (c - o) / hl,
        "KUP": (h - max(o, c)) / (o + EPS),
        "KUP2": (h - max(o, c)) / hl,
        "KLOW": (min(o, c) - l) / (o + EPS),
        "KLOW2": (min(o, c) - l) / hl,
        "KSFT": (2 * c - h - l) / (o + EPS),
        "KSFT2": (2 * c - h - l) / hl,
    }


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def compute_f3(df: pd.DataFrame, i: int) -> dict:
    """봉 i의 F3 피처 20개 (KBAR 9 + 롤링 11, 창 20).

    창(High/Low/Volume 기반) 계산은 nan-safe 함수 또는 유한값 마스킹으로
    부분 결측을 처리한다 — 모듈 docstring 참조.
    """
    o = float(df["Open"].iloc[i])
    h = float(df["High"].iloc[i])
    l = float(df["Low"].iloc[i])
    c = float(df["Close"].iloc[i])

    out = {k: _safe(v) for k, v in _kbar(o, h, l, c).items()}

    d = WINDOW
    lo = max(0, i - d + 1)
    close = df["Close"].iloc[lo:i + 1].to_numpy(dtype=float)
    high = df["High"].iloc[lo:i + 1].to_numpy(dtype=float)
    low = df["Low"].iloc[lo:i + 1].to_numpy(dtype=float)
    vol = df["Volume"].iloc[lo:i + 1].to_numpy(dtype=float)
    n = len(close)

    # RSV: 창 내 저가~고가 사이 종가 위치. nanmin/nanmax로 정지봉의 NaN을
    # 걸러낸다 — 봉 i 자신은 항상 유효하므로 전부-NaN이 되는 일은 없다.
    win_low = np.nanmin(low)
    win_high = np.nanmax(high)
    rsv = (c - win_low) / ((win_high - win_low) + EPS)

    # 거래량은 로그로 변환(로그수익률과 스케일을 맞추는 qlib 관례). NaN은
    # 그대로 전파시켰다가 아래에서 유한한 위치만 골라 쓴다.
    logv = np.log1p(np.maximum(vol, 0.0))

    corr_mask = np.isfinite(close) & np.isfinite(logv)
    if corr_mask.sum() > 1:
        cm, lm = close[corr_mask], logv[corr_mask]
        corr = (np.corrcoef(cm, lm)[0, 1]
                if cm.std() > 0 and lm.std() > 0 else 0.0)
    else:
        corr = 0.0

    # close는 항상 유한하므로 dc는 원래도 NaN이 없다(방어적으로만 취급).
    dc = np.diff(close) / (close[:-1] + EPS)
    dv = np.diff(logv)  # 정지봉과 인접한 위치는 NaN

    cord_mask = np.isfinite(dc) & np.isfinite(dv)
    if cord_mask.sum() > 1:
        dcm, dvm = dc[cord_mask], dv[cord_mask]
        cord = (np.corrcoef(dcm, dvm)[0, 1]
                if dcm.std() > 0 and dvm.std() > 0 else 0.0)
    else:
        cord = 0.0

    dc_valid = dc[np.isfinite(dc)]
    cntp = float((dc_valid > 0).mean()) if dc_valid.size else 0.0
    gains = np.maximum(dc_valid, 0.0).sum()
    total = np.abs(dc_valid).sum()
    sump = gains / (total + EPS)

    # WVMA/VSTD: 거래량 기반이라 NaN이 섞일 수 있다 — nanstd/nanmean으로
    # 유효한 나머지 표본만 쓴다(항상 최소 1개는 유효 → 전부-NaN 경고 없음).
    prod = dc * vol[1:] if len(dc) else np.array([])
    if prod.size and np.isfinite(prod).any():
        wvma = np.nanstd(prod) / (np.nanmean(np.abs(prod)) + EPS)
    else:
        wvma = 0.0
    vstd = np.nanstd(vol) / (np.nanmean(vol) + EPS)

    # BETA/RSQR: close 대 봉 위치(x)의 선형회귀. close는 원래 결측이 없지만
    # 방어적으로 유한한 위치만 회귀에 사용한다(x는 원래 봉 위치를 유지해
    # 결측봉으로 인한 "시간 간격"을 왜곡하지 않는다).
    x = np.arange(n, dtype=float)
    reg_mask = np.isfinite(close)
    if reg_mask.sum() > 1 and close[reg_mask].std() > 0:
        xm, cm2 = x[reg_mask], close[reg_mask]
        slope, intercept = np.polyfit(xm, cm2, 1)
        pred = slope * xm + intercept
        ss_res = float(((cm2 - pred) ** 2).sum())
        ss_tot = float(((cm2 - cm2.mean()) ** 2).sum())
        rsqr = 1.0 - ss_res / (ss_tot + EPS)
    else:
        slope, rsqr = 0.0, 0.0

    # IMAX: 창 내 고가 최댓값의 위치(최근일수록 0에 가깝게). nanargmax로
    # NaN(정지봉)을 최댓값 후보에서 제외한다.
    imax = float(n - 1 - int(np.nanargmax(high))) / d

    qtlu = float(np.nanquantile(close, 0.8)) / (c + EPS)

    out.update({
        f"RSV_{d}": _safe(rsv),
        f"CORR_{d}": _safe(corr),
        f"CORD_{d}": _safe(cord),
        f"CNTP_{d}": _safe(cntp),
        f"SUMP_{d}": _safe(sump),
        f"WVMA_{d}": _safe(wvma),
        f"VSTD_{d}": _safe(vstd),
        f"BETA_{d}": _safe(slope / (c + EPS)),
        f"RSQR_{d}": _safe(rsqr),
        f"IMAX_{d}": _safe(imax),
        f"QTLU_{d}": _safe(qtlu),
    })
    return out
