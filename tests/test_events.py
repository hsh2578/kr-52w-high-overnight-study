# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from surge_d1 import config, events


def make_df(n=300, close_base=100.0, vol=1000.0):
    """i=n-2 지점이 52주 신고가 + 거래량 급증이 되도록 만든 합성 봉."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = np.full(n, close_base)
    v = np.full(n, vol)
    df = pd.DataFrame(
        {"Open": c.copy(), "High": c * 1.01, "Low": c * 0.99,
         "Close": c.copy(), "Volume": v, "halted": False},
        index=idx,
    )
    return df


@pytest.fixture
def big_shares(monkeypatch):
    """모든 코드에 시가총액 1,000억을 넘기는 상장주식수를 준다.

    collect_events는 시점 시가총액을 모르는 종목을 버리므로, 합성 봉으로
    이벤트 수집을 검사하려면 주식수도 같이 넣어줘야 한다."""
    from surge_d1 import sources

    class _Any(dict):
        def get(self, key, default=None):
            return pd.Series([1e10], index=pd.to_datetime(["2019-01-01"]))

    monkeypatch.setattr(sources, "load_list_shares", lambda: _Any())


def test_is_breakout_직전5봉에_거래정지가_있으면_False():
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 128.0     # +28% — 가격제한(0.32) 이내
    df.loc[df.index[i], "Volume"] = 1e9
    assert events.is_breakout(df, i) is True          # 정지 없으면 통과
    df.loc[df.index[i - 2], "halted"] = True
    assert events.is_breakout(df, i) is False


def test_is_breakout_익일이_거래정지면_False():
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 128.0     # +28% — 가격제한(0.32) 이내
    df.loc[df.index[i], "Volume"] = 1e9
    df.loc[df.index[i + 1], "halted"] = True
    assert events.is_breakout(df, i) is False


def test_is_breakout_52주고가를_1_5퍼센트_넘겨야_한다():
    df = make_df()
    i = len(df) - 2
    # 기본 상태: 신고가 아님(전부 동일가) → False
    assert events.is_breakout(df, i) is False

    # 52주 고가(High 101) x 1.015 = 102.515 를 못 넘으면 돌파가 아니다
    df.loc[df.index[i], "Close"] = 102.0
    assert events.is_breakout(df, i) is False

    df.loc[df.index[i], "Close"] = 128.0     # +28% — 상한가(29.5%) 미만
    assert events.is_breakout(df, i) is True

    # 거래량은 조건이 아니다(구 정의에만 있었다)
    df.loc[df.index[i], "Volume"] = 1.0
    assert events.is_breakout(df, i) is True


def test_is_breakout_직전15봉에_이미_넘었으면_신규돌파가_아니다():
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 128.0
    df.loc[df.index[i - 3], "Close"] = 110.0   # 사흘 전에 이미 102.515를 넘음
    assert events.is_breakout(df, i) is False


def test_is_breakout_150일선이_내려가는_중이면_False():
    df = make_df()
    i = len(df) - 2
    # 150일선 창의 앞쪽 20봉만 높게 둔다 → 20일 전 평균이 지금보다 높다(하락 추세)
    df.loc[df.index[i - 169:i - 149], "Close"] = 400.0
    df.loc[df.index[i], "Close"] = 128.0
    assert events.is_breakout(df, i) is False


def test_is_breakout_최소인덱스_미만은_False():
    df = make_df()
    df.loc[df.index[10], "Close"] = 999.0
    df.loc[df.index[10], "Volume"] = 1e9
    assert events.is_breakout(df, 10) is False


def test_labels_시가청산과_종가청산_순수익률():
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 100.0
    df.loc[df.index[i + 1], "Open"] = 105.0
    df.loc[df.index[i + 1], "Close"] = 95.0

    out = events.labels(df, i, cost=0.0021)
    assert out["y_raw"] == pytest.approx(0.05 - 0.0021)
    assert out["y_hold"] == pytest.approx(-0.05 - 0.0021)
    assert out["y_cls"] == 1


def test_labels_손실이면_분류라벨이_0():
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 100.0
    df.loc[df.index[i + 1], "Open"] = 100.1   # +0.1% < 비용 0.21%
    df.loc[df.index[i + 1], "Close"] = 100.0
    assert events.labels(df, i)["y_cls"] == 0


def _write_master(tmp_path):
    """마스터 pkl 픽스처. is_etf 는 일부러 삼성전자까지 True 로 둔다(실제 데이터가 그렇다)."""
    import pickle
    master = pd.DataFrame({
        "종목코드": ["005930", "000810", "000815", "000105"],
        "종목명": ["삼성전자", "삼성화재", "삼성화재우", "유한양행우"],
        "is_common": [True, True, False, False],
        "is_spac": [False, False, False, False],
        "is_reit": [False, False, False, False],
        "is_etf": [True, True, True, False],
    })
    d = tmp_path / "cache"
    d.mkdir()
    with open(d / "krx_master_2026-05-18.pkl", "wb") as f:
        pickle.dump(master, f)
    return d


def _isolated_root(tmp_path):
    """PROJECT_ROOT를 이 경로로 바꾸면 실제 data/universe.json이 보이지 않는다."""
    root = tmp_path / "root"
    root.mkdir()
    return root


def _write_universe(root, excluded, common=None):
    """data/universe.json 픽스처. root는 PROJECT_ROOT로 monkeypatch할 경로."""
    import json
    d = root / "data"
    d.mkdir(parents=True, exist_ok=True)
    payload = {"as_of": "20260827", "common": common or [], "excluded": list(excluded),
               "counts": {}}
    with open(d / "universe.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return root


def test_load_excluded_codes_universe_json이_있으면_그_excluded를_쓴다(monkeypatch, tmp_path):
    root = _write_universe(_isolated_root(tmp_path), excluded=["000815", "0000D0"],
                            common=["005930", "000810"])
    monkeypatch.setattr(events.config, "PROJECT_ROOT", root)
    out = events.load_excluded_codes()
    assert out == {"000815", "0000D0"}


def test_load_excluded_codes_universe_json_없으면_마스터로_폴백한다(monkeypatch, tmp_path):
    monkeypatch.setattr(events.config, "PROJECT_ROOT", _isolated_root(tmp_path))
    monkeypatch.setattr(events.config, "HOLLYKR_CACHE", _write_master(tmp_path))
    out = events.load_excluded_codes()
    # is_etf 를 읽었다면 삼성전자·삼성화재도 제외돼 4건이 됐을 것이다
    assert out == {"000815", "000105"}


def test_load_excluded_codes_둘다_없으면_빈집합_예외없음(monkeypatch, tmp_path):
    monkeypatch.setattr(events.config, "PROJECT_ROOT", _isolated_root(tmp_path))
    empty = tmp_path / "nomaster"
    empty.mkdir()
    monkeypatch.setattr(events.config, "HOLLYKR_CACHE", empty)
    assert events.load_excluded_codes() == set()


def test_load_excluded_codes_어느_목록에도_없는_코드는_제외되지_않는다(monkeypatch, tmp_path):
    """생존편향 가드: 상장폐지 등으로 마스터/유니버스 어느 쪽에도 없는 코드는
    "제외 목록에 있어야만 빠진다" 규칙에 따라 살아남아야 한다."""
    root = _write_universe(_isolated_root(tmp_path), excluded=["000815"],
                            common=["005930"])
    monkeypatch.setattr(events.config, "PROJECT_ROOT", root)
    out = events.load_excluded_codes()
    assert "999999" not in out          # 상장폐지 등 미상 코드는 제외집합에 없어야 한다


def test_collect_events_제외집합에_있으면_빠지고_모르는코드는_남는다(big_shares):
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 128.0     # +28% — 가격제한(0.32) 이내
    df.loc[df.index[i], "Volume"] = 1e9

    # 제외 대상(우선주)
    assert len(events.collect_events({"000815": df}, exclude={"000815"})) == 0
    # 마스터에 없는 신규상장/상장폐지 코드는 생존편향을 줄이므로 남긴다
    assert len(events.collect_events({"999999": df}, exclude={"000815"})) == 1
    # exclude=None 이면 필터를 걸지 않는다
    assert len(events.collect_events({"000815": df}, exclude=None)) == 1


def test_collect_events_universe_json_제외코드는_이벤트에서_빠진다(monkeypatch, tmp_path, big_shares):
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 128.0
    df.loc[df.index[i], "Volume"] = 1e9

    root = _write_universe(_isolated_root(tmp_path), excluded=["069500"])  # ETF 예시
    monkeypatch.setattr(events.config, "PROJECT_ROOT", root)
    ex = events.load_excluded_codes()
    assert len(events.collect_events({"069500": df}, exclude=None)) == 1   # 필터가 없으면 잡힌다
    assert len(events.collect_events({"069500": df}, exclude=ex)) == 0


def test_add_market_adjusted_그날_중앙값을_뺀다():
    ev = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"]),
        "y_exit": [0.01, 0.02, 0.03, 0.05],   # 청산 시점(config.EXIT)에 맞는 라벨
    })
    out = events.add_market_adjusted(ev)
    # 1/2 중앙값 0.02 → -0.01, 0.0, 0.01 / 1/3은 단일이라 0.0
    assert list(out["y_mkt"].round(6)) == [-0.01, 0.0, 0.01, 0.0]


# --- 추가 1: y_hold20 (20봉 보유 수익률) — 이벤트 자격 게이트가 아니다 ---

def test_labels_y_hold20_20봉뒤_수익률을_계산한다():
    df = make_df(n=300)
    i = 200  # i+20 = 220 < 300, 미래 데이터 충분
    df.loc[df.index[i], "Close"] = 100.0
    df.loc[df.index[i + 20], "Close"] = 110.0
    out = events.labels(df, i)
    assert out["y_hold20"] == pytest.approx((110.0 - 100.0) / 100.0 - config.COST)


def test_labels_y_hold20_미래봉없으면_NaN이지만_다른_라벨은_정상():
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 100.0
    df.loc[df.index[i + 1], "Open"] = 105.0
    df.loc[df.index[i + 1], "Close"] = 95.0
    out = events.labels(df, i)
    assert np.isnan(out["y_hold20"])
    assert out["y_raw"] == pytest.approx(0.05 - config.COST)


def test_collect_events_20봉_미래가_없어도_이벤트는_생성되고_y_hold20만_NaN(big_shares):
    """시계열 맨 끝 이벤트를 20봉 미래가 없다고 통째로 버리면 최근 한달치 이벤트가
    전부 사라진다 — 라이브 데모에서 가장 보고 싶은 구간이라 반드시 살려야 한다."""
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 128.0
    df.loc[df.index[i], "Volume"] = 1e9
    ev = events.collect_events({"000001": df}, exclude=None)
    assert len(ev) == 1
    assert np.isnan(ev.loc[0, "y_hold20"])


# --- 추가 2: y_up (비용 미차감 총상승 지표, 집계 전용) ---

def test_labels_y_up은_비용을_빼지_않은_총상승_지표다():
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 100.0
    df.loc[df.index[i + 1], "Open"] = 100.1   # 총상승 +0.1% (비용 0.21%엔 못 미침)
    df.loc[df.index[i + 1], "Close"] = 100.0
    out = events.labels(df, i)
    assert out["y_up"] == 1     # 총상승 기준으로는 상승
    assert out["y_cls"] == 0    # 비용 차감 기준으로는 순손실


# --- 추가 3: KRX 일일 가격제한(±30%, 틱 반올림 여유 포함 0.32) 위반 봉 배제 ---

def test_is_breakout_상한가는_종가에_살_수_없으므로_제외한다():
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 130.1     # +30.1% — 상한가 틱 반올림으로 흔한 값
    assert events.is_breakout(df, i) is False
    df.loc[df.index[i], "Close"] = 129.6     # +29.6% — 경계(29.5%) 바로 위
    assert events.is_breakout(df, i) is False
    df.loc[df.index[i], "Close"] = 129.4
    assert events.is_breakout(df, i) is True


def test_collect_events_시가총액_1000억_미만은_빠진다(monkeypatch):
    from surge_d1 import sources
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 128.0
    small = pd.Series([1e6], index=pd.to_datetime(["2019-01-01"]))   # 시총 1.28억
    monkeypatch.setattr(sources, "load_list_shares", lambda: {"000001": small})
    assert len(events.collect_events({"000001": df}, exclude=None)) == 0


def test_is_breakout_가격제한_35퍼센트는_거부():
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 135.0     # +35% — ±30% 상한가로는 물리적으로 불가능
    df.loc[df.index[i], "Volume"] = 1e9
    assert events.is_breakout(df, i) is False
