# -*- coding: utf-8 -*-
"""F3 — Microsoft qlib Alpha158 계열 피처 20개. 스펙 §5.3.

원 정의: https://github.com/microsoft/qlib — qlib/contrib/data/loader.py 의
Alpha158DL.get_feature_config() (그리고 qlib/data/ops.py의 IdxMax/Std/Corr
등 연산자 정의). KBAR 9종은 원식 그대로, 롤링은 d=20 창의 11종을 옮겼다
(스펙은 d=5,20 두 창을 언급하지만 20개 총원에 맞추려고 **롤링은 d=20
하나만** 쓴다 — KBAR 9 + 롤링 11 = 20). 재현이 목적이므로 정의를
"개선"하지 않는다 — 이상해 보이는 식도 그대로 옮기고 리포트에 기록한다.

리뷰(2026-08-28, 2회에 걸쳐 진행)에서 qlib 원본 소스와 대조해 총 6건의
정의 이탈(치명 2건 + 중요 3건 + 사소 1건, 1차)과 ddof 1건(2차, 사소하지
않음 — 아래 8번)이 발견되어 이 버전에서 수정했다. 상세 근거는
task-6-report.md의 "Fix report" 절 참조. 요약:

1. **IMAX_20 (치명)**: qlib의 IdxMax(High, d)/d는 "창의 가장 오래된 봉을
   1, 오늘을 d로 세는 1-기반 위치"를 반환한다 — 오늘이 창의 최고가면
   IMAX_20=1.0. 이전 구현은 그 정반대(오늘이 최고가면 0.0)를 계산했다.
   이 스펙의 이벤트는 정의상 52주 신고가라 "오늘이 창의 고가"가 흔한
   경우이므로, 이 피처가 뒤집혀 있으면 가장 중요한 신호 하나가 통째로
   반전된다.
2. **VSTD_20 (치명)**: qlib은 `Std($volume,d) / ($volume+1e-12)` — 분모는
   **오늘 하루의 거래량**이지 창 평균 거래량이 아니다. 이전 구현은
   창 평균으로 나눠 실측 상관계수 0.108의 별개 피처가 되어 있었다.
3. **WVMA_20 (중요)**: qlib은 분자·분모 둘 다 **절대수익률**
   `Abs($close/Ref($close,1)-1) * $volume`을 쓴다. 이전 구현은 분자에서만
   부호 있는 수익률을 썼다(분모만 Abs 적용 — 내부적으로 모순).
4. **CORD_20 (중요)**: 거래량 항은 `Log($volume/Ref($volume,1) + 1)`
   (전일 대비 거래량 "비율"의 로그)이다. 이전 구현은 `diff(log1p(V))`
   = `log((1+V_t)/(1+V_{t-1}))`로 다른 함수를 계산했다. 가격 항
   `$close/Ref($close,1)`은 `(C_t-C_{t-1})/C_{t-1}`과 아핀 이동 관계라
   피어슨 상관계수에 영향이 없으므로 그대로 둔다.
5. **SUMP_20 (중요)**: qlib은 `Sum(Greater($close-Ref($close,1),0),d) /
   (Sum(Abs($close-Ref($close,1)),d)+1e-12)` — **원시 가격차**를 쓴다.
   이전 구현은 퍼센트 수익률을 썼다. 창 안에서 각 날의 분모(C_{t-1})가
   다 다르므로 이는 단순 스케일 불변 재조정이 아니다.
6. **d+1 윈도우 (중요)**: CNTP_20/SUMP_20/CORD_20/WVMA_20은 전부 `Ref`로
   한 칸 시프트한 차분 계열의 d개 값을 요구하는데, 차분 계열의 t번째
   값은 원계열의 t, t-1 두 점이 필요하므로 **d+1개의 원시 봉**이
   필요하다. 이전 구현은 20개짜리 창을 슬라이스한 뒤 차분해 19개
   관측치만 만들었다. 리뷰어가 qlib 미설치 상태로 "중간 신뢰도"로
   지적했으나, 코디네이터가 "Mean(X,d)는 X의 d개 값에 대한 롤링 평균이고
   X_t는 close[t],close[t-1] 두 점이 필요하다"는 근거로 d+1이 맞다고
   확정했다 — 그 결론을 그대로 채택했다.
7. **분모의 1e-12 (사소)**: qlib은 "2"-접미사 KBAR 4종의 (High-Low) 분모와
   일부 롤링 분모(RSV의 고저폭, SUMP/WVMA의 절대합/평균, VSTD의 거래량,
   RSQR의 잔차제곱합)에만 1e-12를 더한다. **$open·$close 단일값 분모에는
   더하지 않는다.** 이전 구현은 KMID/KLEN/KUP/KLOW/KSFT와 BETA_20/QTLU_20
   에도 1e-12를 넣었다. 실측 KRW 가격에서는 수치적으로 무의미(~1e-14)
   하지만 "qlib과 동치"라는 이 모듈의 주장과 모순되므로 제거했다. 분모가
   0이거나 비유한이면 1e-12로 얼버무리지 않고 **NaN**을 반환한다 — 이벤트
   봉의 Open·Close는 항상 양수·유한(아래 참조)이라 이 경로는 실데이터에서
   발동하지 않음을 검증했다.
8. **VSTD_20/WVMA_20의 ddof (2차 리뷰, qlib 원본 재구현으로 확인)**:
   qlib의 `Std` 연산자는 `series.rolling(N, min_periods=1).std()`이고
   pandas `.std()`의 기본은 **ddof=1**(표본표준편차)이다. 이전 구현은
   `np.nanstd`(기본 ddof=0, 모표준편차)를 썼다 — 398개 정상봉 이벤트
   기준 최대 절대오차 VSTD_20=0.0717, WVMA_20=0.111(둘 다 ddof=0
   재구성과 정확히 일치해 원인이 확인됨), 대략 sqrt(d/(d-1))≈2.6%의
   체계적 과소추정이었다. `_sample_std()` 헬퍼로 두 피처의 분자만
   ddof=1로 바꿨다(Mean() 항은 ddof가 적용되지 않으므로 그대로).

NaN(거래정지 봉) 처리 방침
--------------------------
sources.mark_halted 규칙: 거래정지 봉은 Open/High/Low/Volume이 NaN이고
Close만 직전값을 유지한다(그리고 남은 정상 봉의 고저 불변식도 복구한다 —
sources.py 참조). 20봉 롤링 창에 정지봉이 섞이는 것은 일상적이다(전체의
약 1.6%). np.corrcoef/np.polyfit/np.quantile/np.argmax/.std()/.mean()을
그대로 쓰면 NaN을 전파하거나 오동작하므로 다음 원칙으로 구현한다:

1. events.is_breakout()은 이벤트 당일(i)을 포함해 i-4..i+1 구간에 정지봉이
   있으면 이벤트로 인정하지 않는다(HALT_LOOKBACK=5). 따라서 **봉 i 자신은
   항상 유효(비정지)** 하다 — compute_f3에 들어오는 o,h,l,c와 오늘의
   거래량은 항상 유한하다. 20봉 창(i-19..i)도 최소 1개(=i 자신)는 항상
   유효한 원소를 갖는다 — "창 전체가 NaN"인 경우는 이 데이터 구조상
   발생하지 않는다.
2. Close는 정지봉에서도 NaN이 아니므로(직전값 유지), Close만 쓰는 계산은
   원래도 NaN에 덜 노출된다. 그래도 방어적으로 nan-safe 버전을 쓴다.
3. High/Low/Volume을 쓰는 계산은 np.nanmin/np.nanmax/np.nanmean/
   np.nanargmax로 바꾸거나, 상관·회귀처럼 두 배열을 함께 쓰는 경우는
   두 배열 모두 유한한 위치만 골라 마스킹한 뒤 계산한다. 표준편차만은
   `np.nanstd`(ddof=0)가 아니라 `_sample_std`(ddof=1, qlib과 동일)를
   쓴다 — 8번 항목 참조.
4. RSV_20은 과거 버그(정지봉 Low=0)가 정확히 이 피처를 오염시켰던 지점이다.
   지금은 정지봉이 Low=0이 아니라 NaN이라 원인은 다르지만, 창 안의 결측을
   min/max에서 제대로 걸러내지 않으면 같은 방식으로 재발한다. nanmin/
   nanmax로 막는다.

_safe()가 0.0으로 접는 이유 (그리고 qlib과의 차이)
----------------------------------------------------
`_safe(x)`는 CORR_20/CORD_20/RSQR_20 등에서 상관·회귀가 수학적으로
정의되지 않는 경우(표준편차 0)와, 부동소수점 연산이 뜻밖의 inf/NaN을
낸 경우를 0.0으로 접는다. **주의: 이는 qlib과 다르다.** qlib의 Corr/
Rsquare 연산자는 표준편차가 0에 가까우면(`np.isclose(std, 0,
atol=2e-05)`) 명시적으로 **NaN**을 반환한다 — 0.0이 아니다. 이 모듈은
의도적으로 0.0을 택했다("상관 없음"을 0으로 나타내는 것이 회귀식에서도
관례적이고, LightGBM 입력으로 NaN보다 다루기 쉽다는 판단). 이 선택이
실제로 문제가 되지 않는 이유는, 이벤트 봉 i가 항상 유효해 20봉 창이
통째로 결측이 되는 경우가 없고, 실측(28,402개 이벤트) 결과 이 0.0-폴백
분기가 단 한 번도 발동하지 않았기 때문이다(§ task-6-report.md 검증 절
참조) — 즉 "qlib과 동일한 관례"라는 주장은 부정확했고 이번에 정정했다.
KMID/KLEN/KUP/KLOW/KSFT와 BETA_20/QTLU_20은 반대로 **NaN을 그대로
반환**한다(위 7번 항목) — _safe를 거치지 않는다.

F1 대비 참고: sources.load_ohlcv가 이제 정상 봉의 High/Low를
max(High,Open,Close)/min(Low,Open,Close)로 복구하므로(스플릿 반올림 오차
정정), RSV_20/KMID2가 [0,1]/[-1,1]을 벗어나던 과거 현상은 더 이상
재현되지 않는다 — 전체 모집단에서 정확히 0/1 경계까지만 관측된다.
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

# $open·$close 단일값으로 나누는 KBAR 5종 — qlib은 이 분모들에 1e-12를
# 더하지 않는다. 분모가 0/비유한이면 NaN을 반환한다(§ 모듈 docstring 7번).
_NO_EPS_KBAR = ("KMID", "KLEN", "KUP", "KLOW", "KSFT")


def _ratio_or_nan(num: float, denom: float) -> float:
    """denom이 0이거나 비유한이면 NaN, 아니면 num/denom.

    qlib이 $open·$close 단일값 분모에 1e-12를 더하지 않는 것과 동일하게
    맞추기 위한 헬퍼 — EPS로 얼버무리는 대신 명시적으로 미정의를 표시한다.
    """
    if np.isfinite(denom) and denom != 0.0:
        return num / denom
    return float("nan")


def _kbar(o: float, h: float, l: float, c: float) -> dict:
    """qlib Alpha158 KBAR 9종.

    o(시가)로 나누는 5종(KMID/KLEN/KUP/KLOW/KSFT)은 1e-12를 더하지 않고
    _ratio_or_nan으로 처리한다. "2"-접미사 4종은 (h-l)+EPS로 나누는데,
    이 분모는 qlib이 실제로 1e-12를 더하는 지점이라 그대로 둔다.
    """
    hl = (h - l) + EPS
    return {
        "KMID": _ratio_or_nan(c - o, o),
        "KLEN": _ratio_or_nan(h - l, o),
        "KMID2": (c - o) / hl,
        "KUP": _ratio_or_nan(h - max(o, c), o),
        "KUP2": (h - max(o, c)) / hl,
        "KLOW": _ratio_or_nan(min(o, c) - l, o),
        "KLOW2": (min(o, c) - l) / hl,
        "KSFT": _ratio_or_nan(2 * c - h - l, o),
        "KSFT2": (2 * c - h - l) / hl,
    }


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def _sample_std(arr: np.ndarray) -> float:
    """qlib의 Std 연산자(`series.rolling(N, min_periods=1).std()`)와 일치하는
    표본표준편차(ddof=1)를, NaN(정지봉)을 걸러낸 유효값만으로 계산한다.

    pandas의 기본 `.std()`는 ddof=1(표본표준편차)이다. 이전 구현은
    `np.nanstd`(기본 ddof=0, 모표준편차)를 써서 VSTD_20/WVMA_20이 qlib
    대비 체계적으로 약 sqrt(d/(d-1))약2.6% 과소추정됐다(2026-08-28 리뷰,
    398개 정상봉 이벤트 기준 최대 절대오차 VSTD_20=0.0717,
    WVMA_20=0.111 - 둘 다 ddof=0 재구성과 정확히 일치해 원인이 확인됨).
    이 헬퍼로 두 피처의 분자(표준편차)만 고친다 - Mean() 항은 ddof가
    적용되지 않으므로 그대로 둔다.

    유효값이 2개 미만이면 ddof=1 표준편차가 수학적으로 정의되지 않는다
    (분산의 분모 n-1=0). qlib 자신도 `rolling(N, min_periods=1).std()`가
    관측치 1개짜리 창에서 NaN을 반환한다 - 그 관례를 그대로 따라 여기서도
    NaN을 반환한다(np.nanstd(arr, ddof=1)을 유효값 1개에 직접 쓰면 같은
    NaN이 나오지만 "Degrees of freedom <= 0" RuntimeWarning이 함께
    발생하므로, 미리 걸러 명시적으로 NaN을 반환해 경고 없이 같은 결과를
    낸다 - F2의 gap_vol_20d와 같은 관례). 이 NaN은 호출부에서 `_safe()`를
    거쳐 0.0으로 접힌다 - CORR_20/RSQR_20/CORD_20의 "표준편차 0" 분기와
    동일하게, qlib의 NaN 관례에서 의도적으로 벗어난 것이다(리포트 참조).
    실측(28,402개 이벤트)에서는 이벤트 봉 i가 항상 유효해 유효값이 2개
    미만인 창이 한 번도 나오지 않았다 - 아래 검증에서 확인한다.
    """
    valid = arr[np.isfinite(arr)]
    if valid.size >= 2:
        return float(valid.std(ddof=1))
    return float("nan")


def compute_f3(df: pd.DataFrame, i: int) -> dict:
    """봉 i의 F3 피처 20개 (KBAR 9 + 롤링 11, 창 20).

    두 종류의 창을 쓴다:
      - "d 창"(lo..i, 최대 20개 원시 봉): 레벨 기반 피처
        (RSV/CORR/VSTD/BETA/RSQR/IMAX/QTLU)에 쓴다.
      - "d+1 창"(lo_e..i, 최대 21개 원시 봉): 차분(Ref 시프트) 기반 피처
        (CNTP/SUMP/CORD/WVMA)에 쓴다 — 차분 계열의 d개 값을 만들려면
        원시 값이 d+1개 필요하다(모듈 docstring 6번).
    """
    o = float(df["Open"].iloc[i])
    h = float(df["High"].iloc[i])
    l = float(df["Low"].iloc[i])
    c = float(df["Close"].iloc[i])
    v_today = float(df["Volume"].iloc[i])
    # 이벤트 봉은 events.is_breakout이 정지를 배제하므로 항상 유효해야
    # 한다 — VSTD_20 분모가 "오늘 거래량"이므로 명시적으로 확인한다
    # (리뷰 Critical 2: "Bar i is never halted ... but assert that").
    # BETA_20/QTLU_20의 분모(오늘 종가)는 반대로 assert로 강제하지 않고
    # _ratio_or_nan류 가드로 NaN을 반환하게 둔다 — 그래야 그 가드 경로를
    # 단위 테스트로 직접 검증할 수 있고, 실데이터에서 항상 유한·양수라는
    # 사실은 전체 모집단 검증(task-6-report.md)에서 별도로 확인한다.
    assert np.isfinite(v_today) and v_today > 0, "이벤트 봉의 거래량이 비정상(정지 배제 가정 위반)"

    kbar = _kbar(o, h, l, c)
    out = {k: (float(kbar[k]) if k in _NO_EPS_KBAR else _safe(kbar[k]))
           for k in KBAR_NAMES}

    d = WINDOW
    # --- "d 창": 레벨 기반 피처 ---------------------------------------
    lo = max(0, i - d + 1)
    close_d = df["Close"].iloc[lo:i + 1].to_numpy(dtype=float)
    high_d = df["High"].iloc[lo:i + 1].to_numpy(dtype=float)
    low_d = df["Low"].iloc[lo:i + 1].to_numpy(dtype=float)
    vol_d = df["Volume"].iloc[lo:i + 1].to_numpy(dtype=float)
    n_d = len(close_d)

    # RSV: 창 내 저가~고가 사이 종가 위치. nanmin/nanmax로 정지봉의 NaN을
    # 걸러낸다 — 봉 i 자신은 항상 유효하므로 전부-NaN이 되는 일은 없다.
    win_low = np.nanmin(low_d)
    win_high = np.nanmax(high_d)
    rsv = (c - win_low) / ((win_high - win_low) + EPS)

    # CORR_20 = Corr($close, Log($volume+1), d) — 레벨 기반, d 창.
    logv_d = np.log1p(np.maximum(vol_d, 0.0))
    corr_mask = np.isfinite(close_d) & np.isfinite(logv_d)
    if corr_mask.sum() > 1:
        cm, lm = close_d[corr_mask], logv_d[corr_mask]
        corr = (np.corrcoef(cm, lm)[0, 1]
                if cm.std() > 0 and lm.std() > 0 else 0.0)
    else:
        corr = 0.0

    # BETA/RSQR: close 대 봉 위치(x)의 선형회귀, d 창.
    x = np.arange(n_d, dtype=float)
    reg_mask = np.isfinite(close_d)
    if reg_mask.sum() > 1 and close_d[reg_mask].std() > 0:
        xm, cm2 = x[reg_mask], close_d[reg_mask]
        slope, intercept = np.polyfit(xm, cm2, 1)
        pred = slope * xm + intercept
        ss_res = float(((cm2 - pred) ** 2).sum())
        ss_tot = float(((cm2 - cm2.mean()) ** 2).sum())
        rsqr = 1.0 - ss_res / (ss_tot + EPS)
    else:
        slope, rsqr = 0.0, 0.0
    beta = slope / c if (np.isfinite(c) and c != 0.0) else float("nan")

    # IMAX = IdxMax($high, d) / d — qlib은 창의 가장 오래된 봉을 1, 가장
    # 최근(오늘) 봉을 d로 세는 1-기반 위치를 쓴다. nanargmax로 NaN(정지봉)을
    # 최댓값 후보에서 제외하고, 0-기반 위치에 +1해 1-기반으로 바꾼다.
    idx_from_oldest_1based = int(np.nanargmax(high_d)) + 1
    imax = float(idx_from_oldest_1based) / d

    qtlu_raw = float(np.nanquantile(close_d, 0.8))
    qtlu = qtlu_raw / c if (np.isfinite(c) and c != 0.0) else float("nan")

    # VSTD = Std($volume, d) / ($volume + 1e-12) — 분모는 "오늘"의
    # 거래량이지 창 평균이 아니다. 분자(창 표준편차)는 qlib과 동일하게
    # 표본표준편차(ddof=1)를 쓰고, NaN(정지봉)은 _sample_std가 걸러낸다.
    vstd = _sample_std(vol_d) / (v_today + EPS)

    # --- "d+1 창": 차분(Ref 시프트) 기반 피처 --------------------------
    lo_e = max(0, i - d)
    close_e = df["Close"].iloc[lo_e:i + 1].to_numpy(dtype=float)
    vol_e = df["Volume"].iloc[lo_e:i + 1].to_numpy(dtype=float)

    # 가격항: $close/Ref($close,1) - 1 = (C_t-C_{t-1})/C_{t-1}. Close는
    # 항상 유한·양수이므로(정지 배제) EPS 없이 나눈다.
    price_ret = (close_e[1:] - close_e[:-1]) / close_e[:-1]
    raw_diff = close_e[1:] - close_e[:-1]  # $close - Ref($close,1) (SUMP/CNTP용 원시 가격차)

    # 거래량항(CORD 전용): Log($volume/Ref($volume,1) + 1). vol_e에 NaN이
    # 있으면(정지봉) 비율도 NaN → 아래 마스킹에서 자연히 제외된다.
    vol_ratio = vol_e[1:] / vol_e[:-1]
    logv_ratio = np.log1p(vol_ratio)

    cord_mask = np.isfinite(price_ret) & np.isfinite(logv_ratio)
    if cord_mask.sum() > 1:
        pm, lvm = price_ret[cord_mask], logv_ratio[cord_mask]
        cord = (np.corrcoef(pm, lvm)[0, 1]
                if pm.std() > 0 and lvm.std() > 0 else 0.0)
    else:
        cord = 0.0

    diff_valid = raw_diff[np.isfinite(raw_diff)]
    cntp = float((diff_valid > 0).mean()) if diff_valid.size else 0.0
    gains = np.maximum(diff_valid, 0.0).sum()
    total = np.abs(diff_valid).sum()
    sump = gains / (total + EPS)

    # WVMA = Std(Abs(ret)*volume, d) / (Mean(Abs(ret)*volume, d) + 1e-12).
    # 분자·분모 둘 다 절대수익률을 쓴다(qlib 원식) — 부호 있는 수익률을
    # 섞지 않는다. 분자(표준편차)는 qlib과 동일하게 ddof=1을 쓴다(분모의
    # Mean()은 ddof가 적용되지 않으므로 nanmean 그대로).
    prod_abs = np.abs(price_ret) * vol_e[1:]
    if prod_abs.size and np.isfinite(prod_abs).any():
        wvma = _sample_std(prod_abs) / (np.nanmean(prod_abs) + EPS)
    else:
        wvma = 0.0

    out.update({
        f"RSV_{d}": _safe(rsv),
        f"CORR_{d}": _safe(corr),
        f"CORD_{d}": _safe(cord),
        f"CNTP_{d}": _safe(cntp),
        f"SUMP_{d}": _safe(sump),
        f"WVMA_{d}": _safe(wvma),
        f"VSTD_{d}": _safe(vstd),
        f"BETA_{d}": float(beta),
        f"RSQR_{d}": _safe(rsqr),
        f"IMAX_{d}": _safe(imax),
        f"QTLU_{d}": float(qtlu),
    })
    return out
