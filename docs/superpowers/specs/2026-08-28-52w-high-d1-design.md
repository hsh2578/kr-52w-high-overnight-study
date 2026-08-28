# 52주 신고가 돌파의 익일(D+1) 초과수익 예측 — 설계 문서

작성일: 2026-08-28 (개정 2: 선행연구·오픈소스 반영)
목적: 학회 발표 (발표까지 약 1주)

## 1. 연구 질문

**52주 신고가를 돌파한 종목을 종가에 매수해 익일 시가에 매도하면 초과수익이 있는가?
있다면 사전에 예측할 수 있는가? 예측 지평(D+1)에 맞춰 피처를 재설계하면, 그리고
재료(catalyst) 정보를 더하면 각각 얼마나 개선되는가?**

기존 두 프로젝트를 합치고, 선행연구와 오픈소스에서 검증된 요소를 추가한다.

| 출처 | 가져오는 것 |
|---|---|
| `주식웹사이트/stock-screener-kr` | 52주 신고가 돌파 이벤트 정의, 기술 피처 14개, LightGBM 파이프라인 |
| `모의투자대회` (catalyst_engine) | D+1 시가청산 지평, 왕복비용 모델, 이원 백테스트, 재료유형 prior, TF-IDF 유사사례 |
| Microsoft **qlib** Alpha158 | KBAR 9종 + 롤링 팩터군 (§5.3) |
| **López de Prado** (2018) | meta-labeling 프레이밍, purged CV + embargo, 표본 고유성 가중 |
| **Cooper·Cliff·Gulen**, **George·Hwang** | 이론적 근거 (§2) |
| 신규 | walk-forward 검증, 확률 보정, 누수 편향 정량화, 5단 ablation |

## 2. 선행연구 — 이 연구가 서 있는 자리

### 2.1 오버나이트 수익률 이상현상

Cooper, Cliff & Gulen, *Return Differences between Trading and Non-Trading Hours:
Like Night and Day* — 미국 주식 리스크 프리미엄의 대부분이 **장 마감~다음 개장 사이**에
발생한다. 후속 연구에서 **모멘텀 전략의 수익은 전적으로 오버나이트에 누적되고, 다른
전략의 수익은 전적으로 장중에 발생한다**는 결과가 보고되었다.

이는 우리 실측과 정확히 일치한다. 52주 신고가 돌파는 모멘텀 전략이고, 우리 데이터에서
시가청산(+0.335%)이 종가청산(+0.110%)보다 3배 크다. 즉 이 프로젝트의 결과는 고립된
데이터 마이닝이 아니라 **기존 문헌의 예측을 한국 시장에서 확인하는 것**이다. 발표에서
이 연결을 명시한다.

### 2.2 52주 신고가 모멘텀

George & Hwang (2004), *The 52-Week High and Momentum Investing*, Journal of Finance.
현재가의 52주 신고가 대비 근접도가 기존 모멘텀 지표(6·12개월 수익률)보다 미래 수익
예측력이 높으며, 그 수익은 장기적으로 반전되지 않는다.

이벤트를 "52주 신고가 돌파"로 잡은 것의 근거다.

### 2.3 Meta-labeling

López de Prado (2018). 1차 모델이 **방향(side)** 을 정하고, 2차 ML 모델이 **실행 여부와
베팅 크기**를 정하는 2단 구조.

**우리 설계가 그대로 meta-labeling이다.** 52주 신고가 돌파 룰 = 1차(side, 매수),
M1~M4 = 2차(이 돌파를 실제로 살 것인가의 확률). 이 프레이밍을 명시하면 "왜 룰로 거른 뒤
ML을 쓰나"라는 질문에 문헌으로 답할 수 있다.

### 2.4 검증 방법론

López de Prado의 **purged cross-validation + embargo**. 시계열에서 무작위 K-fold는
미래정보 누수를 일으킨다. 라벨이 미래 구간에 걸치면 그 구간을 학습에서 제거(purge)하고,
검정 구간 직후를 일정 기간 배제(embargo)한다.

우리 라벨은 D+1이라 purge 필요성은 작지만, **같은 날 여러 종목 이벤트가 시장 전체
충격을 공유**하므로 embargo와 표본 고유성 가중이 필요하다(§7.3).

## 3. 사전 실측 (설계 근거)

데이터: HollyKR OHLCV 캐시 `.cache/ohlcv/ohlcv_cache.pkl` — 1,978종목, ~2026-05-18,
종목당 최대 1,500거래일. 읽기 전용.

| 이벤트 (거래대금 100억+) | n | 익일 시가청산 | 승률 | t값 | 익일 종가청산 |
|---|---|---|---|---|---|
| **52주 신고가 돌파 + 거래량 1.5배** | 22,062 | **+0.335%** | 43.5% | **14.51** | +0.110% (t=2.54) |
| 당일 급등주 +12~29.8% | 10,609 | +0.228% | 43.3% | 5.96 | −0.279% (t=−3.93) |

왕복비용 0.21% 차감. 위 22,062는 `i>=252` 기준이며 최종 데이터셋은 `i>=260`을 써 21,821건.

**연도별 (레짐 의존성 — 핵심 결과 중 하나):**

| 연도 | n | 익일 시가청산 평균 |
|---|---|---|
| 2020 | 615 | +0.360% |
| 2021 | 3,486 | +0.216% |
| 2022 | 1,276 | +0.075% |
| 2023 | 3,259 | +0.279% |
| 2024 | 3,101 | **+0.002%** |
| 2025 | 5,039 | +0.315% |
| 2026 (1~5월) | 5,045 | **+0.745%** |

전체 평균은 안정적 상수가 아니다. 2024년엔 알파가 사라지고 2026년엔 3배가 된다.
**walk-forward가 필수인 이유가 데이터로 증명된다** — 무작위 분할은 이 구조를 은폐한다.

**재료 매칭률:** 52주 신고가 돌파 17,820건 중 1,781건(10.0%). 급등주는 4,854건 중
2,548건(52.5%). 재료는 보조 분석으로 두고 교란은 `day_change` 통제로 완화한다.

## 4. 데이터 갱신 (D0 — 필수 선행 작업)

현재 캐시는 **2026-05-18**에서 끊겨 있다(오늘 기준 102일 갭). 그대로 쓰면:

1. 2026년 데이터를 절반 버린다 — 가장 강한 레짐(+0.745%)의 6~8월이 없다.
2. 재료 CSV 880행이 사장된다 — 재료는 2026-07-23까지 있는데 OHLCV가 먼저 끊겨 라벨을 못 붙인다.
3. `predict.py`가 오늘 못 돈다 — 발표 데모 불가.

**조치:**

- FinanceDataReader(설치 확인: 0.9.101)로 기존 1,978종목 **증분 갱신**(예상 ~5분). 필수.
- 미커버 종목(전체 약 2,700 중 722종목) **신규 수집**은 백그라운드로 병행. 시간 내 완료되면
  포함, 아니면 1,978종목으로 진행. 어느 쪽이든 리포트에 커버리지를 명시한다.
- 재료 CSV는 신규 수집하지 않는다(OCR 재수집은 수일 소요, §12 범위 밖).

## 5. 피처 설계

기존 14개를 그대로 쓰지 않는다. 그 피처들은 **20일 내 최고 수익률** 예측용으로 설계된
것이라, D+1로 지평이 바뀌면 상당수가 무의미해진다(`base_length`, `days_since_ath`,
`ma200_slope`, `volatility_contraction`, `volume_dry_up` = 베이스 형성 품질 = 20일 개념).
그래서 **피처군을 4개로 나누고 단계적으로 쌓아 기여도를 측정한다.**

### 5.1 F1 — 기존 기술 피처 14개 (재현용)

`stock-screener-kr/ml/src/collect_training_data.py` 의 `calculate_features()` 를 그대로 이식.

`breakout_pct`, `volume_surge`, `close_strength`, `base_length`,
`volatility_contraction`, `volume_dry_up`, `ma200_slope`, `pct_above_52w_low`,
`rs_vs_market`, `market_return`, `ma20_deviation`, `liquidity`, `days_since_ath`,
`atr_ratio`

### 5.2 F2 — D+1 특화 피처 10개 (신규 설계)

익일 **시가**를 맞히는 데 직접 관련된 것들. 기존 피처군에 전혀 없던 축이다.

| 피처 | 정의 | 근거 |
|---|---|---|
| `avg_gap_20d` | 최근 20일 이 종목의 평균 시가 갭 `(O[t]-C[t-1])/C[t-1]` | **종목별 갭 성향.** D+1 시가 예측에 가장 직접적 |
| `gap_vol_20d` | 위 갭의 표준편차 | 갭 예측 불확실성 |
| `gap_open` | 당일 시가 갭 | 당일 갭의 지속/반전 |
| `day_change` | 당일 등락률 | **교란 통제변수** (재료 유무가 당일 급등과 얽힘) |
| `intraday_range` | `(H-L)/C[t-1]` | 당일 변동성 |
| `upper_shadow` | `(H-C)/(H-L)` | 윗꼬리 = 마감 약세 = 익일 갭업 실패 신호 |
| `market_gap_prev` | 직전일 시장 전체 평균 갭 | 레짐 프록시 |
| `dow` | 요일 (0~4) | 주말 갭 리스크(금요일) |
| `aligned` | `C > MA20 > MA60` (0/1) | 모의투자대회 `backtest.py` 정배열 factor |
| `prior_surge_5d` | 직전 5봉 중 +12% 이상 존재 (0/1) | 첫 돌파 vs 연속 급등 |

### 5.3 F3 — qlib Alpha158 계열 20개 (오픈소스 이식)

Microsoft **qlib** 의 `Alpha158` 피처 정의(`qlib/contrib/data/loader.py`)에서 D+1에
관련도가 높은 것을 선별 이식한다. 158개 전부는 1주일 일정에 과하므로 아래 20개로 한정.

**KBAR 9종** — 당일 캔들 형태를 정규화한 것. 마감 위치가 익일 갭에 직결되므로 최우선.

```
KMID  = (close-open)/open              KMID2 = (close-open)/(high-low)
KLEN  = (high-low)/open
KUP   = (high-max(open,close))/open    KUP2  = KUP 분모를 (high-low)로
KLOW  = (min(open,close)-low)/open     KLOW2 = KLOW 분모를 (high-low)로
KSFT  = (2*close-high-low)/open        KSFT2 = KSFT 분모를 (high-low)로
```

**롤링 11종** (윈도우 d = 5, 20):

| 피처 | 정의 |
|---|---|
| `RSV_d` | `(close-min(low,d))/(max(high,d)-min(low,d))` — 스토캐스틱 위치 |
| `CORR_d` | `corr(close, log(volume), d)` — 가격·거래량 동조 |
| `CORD_d` | `corr(close/close[-1], log(volume/volume[-1]), d)` — 변화율 동조 |
| `CNTP_d` | 상승일 비율 |
| `SUMP_d` | `Σmax(Δclose,0) / Σ abs(Δclose)` — RSI류 |
| `WVMA_d` | 거래량 가중 수익률 변동성 |
| `VSTD_d` | 거래량 표준편차 / 거래량 |
| `BETA_d` | 종가 회귀 기울기 / close |
| `RSQR_d` | 종가 회귀 R2 — 추세 직선성 |
| `IMAX_d` | 최고가 이후 경과일 / d |
| `QTLU_d` | `Quantile(close, d, 0.8)/close` |

**왜 이걸 쓰는가**: 우리가 임의로 만든 피처가 아니라 **공개·검증된 팩터 라이브러리**라
발표에서 "피처를 자의적으로 골랐다"는 비판을 방어할 수 있다. 그리고 F1(프로젝트 고유
피처) 대비 **범용 팩터의 상대 기여도**를 측정할 수 있다.

### 5.4 F4 — 재료 피처 5개

`모의투자대회/catalyst_engine/baserate.py`, `schema.py` 이식.

`type_prior`(재료유형 → 실측 D+1 평균), `analog_mean_d1`, `analog_win_rate`, `analog_n`
(사유 텍스트 TF-IDF 최근접 top-10 집계), `reason_len`

**누수 방지(필수)**: `type_prior` 와 TF-IDF 인덱스는 **각 폴드의 학습 구간 데이터만으로**
생성한다. 전체 기간으로 만든 `CATALYST_TYPE_PRIOR` 상수를 그대로 쓰면 미래정보 누수이므로
상수는 참고용으로만 두고 폴드별로 재계산한다.

**합계: F1(14) + F2(10) + F3(20) + F4(5) = 49개**

## 6. 라벨

```
y_raw  = (Open[i+1] - Close[i]) / Close[i] - 0.0021     # 주 라벨: 익일 시가청산 순수익률
y_mkt  = y_raw - (그날 전체 이벤트의 y_raw 중앙값)        # 시장조정 라벨
y_cls  = 1 if y_raw > 0 else 0                          # 분류 라벨
y_hold = (Close[i+1] - Close[i]) / Close[i] - 0.0021     # 보조: 익일 종가청산
```

**`y_mkt` 를 넣는 이유**: 익일 시가 수익률은 시장 전체의 갭에 크게 좌우된다. 횡단면
정규화(cross-sectional normalization)는 qlib을 포함한 표준 관행이며, 이를 병행해야
"시장이 갭업해서 오른 것"과 "이 종목이 특별해서 오른 것"을 분리할 수 있다.
메인 결과는 `y_raw`(실제 손익)로 보고하되, `y_mkt` 결과를 함께 실어 알파의 실체를 검증한다.

## 7. 모델과 실험 설계

### 7.1 5단 ablation

| | 피처 | 개수 | 답하는 질문 |
|---|---|---|---|
| **M0** | 없음 (학습 구간 평균) | 0 | 귀무가설 |
| **M1** | F1 | 14 | *기존 프로젝트를 D+1로 그냥 옮기면?* |
| **M2** | F1+F2 | 24 | *지평에 맞게 피처를 재설계하면?* |
| **M3** | F1+F2+F3 | 44 | *검증된 범용 팩터(qlib)를 더하면?* |
| **M4** | F1+F2+F3+F4 | 49 | *재료를 더하면?* |

M4는 재료가 붙은 ~1,781건에서만 학습 가능하므로, **동일 표본에서 M3를 재학습한 `M3prime`**
와 비교한다. M3 vs M4 직접 비교는 표본이 달라 하지 않는다.

알고리즘은 LightGBM 고정(회귀 `LGBMRegressor`, 분류 `LGBMClassifier`). 모델 탐색은 이
연구의 주제가 아니며, 피처군의 기여를 측정하는 것이 목적이다.

### 7.2 Walk-forward + embargo

M0~M3 — 연 단위 4폴드(expanding window):

| 폴드 | 학습 | 검정 |
|---|---|---|
| 1 | 2020–2022 | 2023 |
| 2 | 2020–2023 | 2024 |
| 3 | 2020–2024 | 2025 |
| 4 | 2020–2025 | 2026 |

M3prime / M4 — 재료 구간이 짧아 단일 폴드: 2024-11~2025-09 학습 → 2025-10~ 검정.

**Embargo 5거래일**: 학습 구간 끝과 검정 구간 시작 사이 5거래일을 버린다. D+1 라벨이라
purge 필요성은 작지만, 같은 시장 충격을 공유하는 인접 표본을 차단한다.

### 7.3 표본 고유성 가중

같은 날 발생한 이벤트들은 시장 전체 갭이라는 **공통 요인을 공유**하므로 독립 표본이
아니다. 어떤 날은 이벤트가 100건, 어떤 날은 3건인데 가중치가 같으면 특정 날짜가 학습을
지배한다.

각 표본에 `weight = 1 / (그날의 이벤트 수)` 를 부여한다(López de Prado의 표본 고유성
개념을 날짜 클러스터에 적용). 가중 적용/미적용 결과를 함께 보고한다.

### 7.4 확률 보정

이 연구의 산출물은 "확률"이므로 분류기 출력이 실제 빈도와 맞아야 한다. LightGBM의 원
출력은 잘 보정되어 있지 않다.

학습 구간에서 **isotonic regression**으로 보정하고, 검정 구간에서 **Brier score** 와
**calibration curve(신뢰도 다이어그램)** 로 보정 전후를 비교한다. 발표 figure로도 쓴다.

### 7.5 지표

- 분류: AUC, **Brier score**, precision@top-10%
- 회귀: MAE, **Spearman IC**(일별 IC의 평균과 t값 = IR)
- **경제적 지표(메인)**: 예측 상위 10% 픽의 평균 순수익률 vs 전체 평균
- 통계검정: bootstrap 95% CI(1,000회), permutation test(1,000회, H0 = 해당 피처군 무의미)
- **다중검정 보정**: 5개 모델 × 2개 라벨을 비교하므로 Benjamini–Hochberg로 p값 보정.
  "여러 번 시도해서 하나 걸린 것 아니냐"는 질문을 선제 방어한다.

### 7.6 누수 편향 정량화 (방법론 기여)

기존 `ml/src/final_model.py:74` 는 `KFold(shuffle=True)` — 시계열에 무작위 셔플 CV는
미래정보 누수다. 동일 데이터를 셔플 CV와 walk-forward 양쪽으로 평가해 **AUC·IC 차이를
측정·보고한다.** §3의 레짐 의존성 표가 이 편향이 왜 생기는지를 설명해 준다.

## 8. 산출물

```
급등주 분석프로그램/
├── surge_d1/
│   ├── sources.py        # OHLCV·universe·재료CSV 로더 + FDR 증분 갱신
│   ├── events.py         # 52주 신고가 돌파 스크린 + 라벨 4종
│   ├── features_f1.py    # 기존 14개
│   ├── features_f2.py    # D+1 특화 10개
│   ├── features_f3.py    # qlib Alpha158 계열 20개
│   ├── features_f4.py    # 재료 5개 (폴드별 point-in-time)
│   ├── dataset.py        # 이벤트 × 피처 × 라벨 → parquet
│   ├── validate.py       # walk-forward + embargo + 가중 + 보정 + 통계검정
│   ├── models.py         # M0~M4 학습·평가
│   └── report.py         # figure + 마크다운 리포트
├── predict.py            # 오늘자 신고가 돌파 종목 → 보정된 확률·기대수익 CSV
├── tests/
├── outputs/figures/
└── docs/연구리포트.md
```

**figure 7장:**

1. D+1 수익률 분포 (52주 신고가 vs 급등주, 시가청산 vs 종가청산)
2. **연도별 기대값** — 레짐 의존성
3. M0 → M1 → M2 → M3 → M4 성능 사다리 (AUC · IC · 상위 decile 수익률)
4. 피처군별 기여도 + permutation test 분포
5. 피처 중요도 (LightGBM gain, F1/F2/F3/F4 색상 구분)
6. **셔플 CV vs walk-forward 편향 크기**
7. **확률 보정 곡선** (보정 전/후)

## 9. 일정 (6일)

| 일 | 작업 | 완료 판정 기준 |
|---|---|---|
| **D0** | FDR 증분 갱신 + 미커버 종목 백그라운드 수집 | 캐시 최신일이 직전 거래일과 일치 |
| D1 | `sources.py`, `events.py`, 라벨 4종 | 이벤트 수 21,821 ±2%(갱신 전 기준), 평균 +0.335% ±0.02%p **재현**, 연도별 표 재현 |
| D2 | `features_f1.py`, `features_f2.py` | 24개 피처 단위 테스트 통과, 결측률 5% 미만 |
| D3 | `features_f3.py`, `features_f4.py` | qlib 공식과 수치 일치 테스트, 폴드별 TF-IDF가 학습 구간만 참조함을 테스트로 검증 |
| D4 | `validate.py`, `models.py` | M0~M4 walk-forward 지표 산출, 보정 적용 |
| D5 | 통계검정 + figure 7장 | png 7개, bootstrap CI · permutation p · BH 보정 p 산출 |
| D6 | 리포트 + `predict.py` | 오늘자 실행 성공, 리포트 완성 |

## 10. 예상 결과와 대응

| 시나리오 | 대응 |
|---|---|
| M1이 M0를 못 이김 | **그 자체가 결과** — "기존 20일용 피처는 D+1에 무용"을 실증 |
| M2/M3가 개선 | 주 결과. "지평에 맞춘 피처 재설계가 핵심" |
| M4(재료)가 유의하지 않음 | null result로 정직 보고. 메인 결과는 이미 확보 |
| 모든 모델이 M0 근처 | "이벤트 자체에 알파가 있고 종목 선별은 불가"라는 결론. §3 t=14.5는 그대로 유효 |

어느 경로로 가도 발표할 결과가 남는다.

## 11. 한계 (발표에 명시)

- OHLCV 커버리지가 1,978 / 약 2,700종목(74%). 소형주 편향 가능. 갱신 후 수치를 재보고한다.
- 기대값이 **레짐 의존적**(2024 +0.002% ~ 2026 +0.745%). 전체 평균만 제시하면 오도된다.
- 재료 데이터는 1.5년 · 매칭률 10%(1,781건). M4 결론은 잠정적이다.
- 재료 표본은 "그날 특징주 TOP20"이라 당일 급등과 얽혀 있다. `day_change` 통제로 완화하되
  완전히 제거되지 않는다.
- 재료유형은 키워드 근사 분류이며 사유는 OCR 전사물이다.
- 익일 시가는 KRX 09:00 일봉 시가 근사. NXT 08:00 개장가 미반영.
- 동시호가 슬리피지는 왕복 0.21%보다 클 수 있다. 특히 상위 decile 픽은 유동성이 몰려
  실제 체결가가 불리할 수 있다.
- 공매도 제약·거래정지 등 실행 제약은 모형화하지 않았다.

## 12. 범위 밖

- 웹 대시보드, GitHub Actions 자동화, 배포
- 텔레그램 발송, 비중 산정, 서킷브레이커 등 실전 운영 기능
- LLM API 호출 (재료는 기존 CSV의 키워드 분류를 사용)
- 재료 데이터 신규 수집 (2026-07 이후 공백은 메우지 않음)
- 모델 아키텍처 탐색 (LightGBM 고정) — 딥러닝 · 강화학습 미포함
- qlib 전체 158개 피처 (선별 20개만)

## 참고문헌

- Cooper, M., Cliff, M., Gulen, H. *Return Differences between Trading and Non-Trading Hours: Like Night and Day.*
- George, T. J., Hwang, C.-Y. (2004). *The 52-Week High and Momentum Investing.* Journal of Finance.
  https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2004.00695.x
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
  (purged CV, embargo, meta-labeling, 표본 고유성)
- Microsoft qlib — Alpha158 피처 정의: https://github.com/microsoft/qlib
  (`qlib/contrib/data/loader.py`)
