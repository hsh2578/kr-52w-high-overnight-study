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


def test_is_breakout_직전5봉에_거래정지가_있으면_False():
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 200.0
    df.loc[df.index[i], "Volume"] = 1e9
    assert events.is_breakout(df, i) is True          # 정지 없으면 통과
    df.loc[df.index[i - 2], "halted"] = True
    assert events.is_breakout(df, i) is False


def test_is_breakout_익일이_거래정지면_False():
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 200.0
    df.loc[df.index[i], "Volume"] = 1e9
    df.loc[df.index[i + 1], "halted"] = True
    assert events.is_breakout(df, i) is False


def test_is_breakout_신고가와_거래량과_거래대금을_모두_요구한다():
    df = make_df()
    i = len(df) - 2
    # 기본 상태: 신고가 아님(전부 동일가) → False
    assert events.is_breakout(df, i) is False

    # 신고가 + 거래량 2배 + 거래대금 충분
    df.loc[df.index[i], "Close"] = 200.0
    df.loc[df.index[i], "Volume"] = 1e9      # 200 * 1e9 = 2e11 > 1e10
    assert events.is_breakout(df, i) is True

    # 거래량만 부족하면 False
    df.loc[df.index[i], "Volume"] = 1000.0
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


def test_load_excluded_codes_우선주만_제외집합에_넣는다(monkeypatch, tmp_path):
    monkeypatch.setattr(events.config, "HOLLYKR_CACHE", _write_master(tmp_path))
    out = events.load_excluded_codes()
    # is_etf 를 읽었다면 삼성전자·삼성화재도 제외돼 4건이 됐을 것이다
    assert out == {"000815", "000105"}


def test_load_excluded_codes_마스터가_없으면_빈집합(monkeypatch, tmp_path):
    empty = tmp_path / "nomaster"
    empty.mkdir()
    monkeypatch.setattr(events.config, "HOLLYKR_CACHE", empty)
    assert events.load_excluded_codes() == set()


def test_collect_events_제외집합에_있으면_빠지고_모르는코드는_남는다():
    df = make_df()
    i = len(df) - 2
    df.loc[df.index[i], "Close"] = 200.0
    df.loc[df.index[i], "Volume"] = 1e9

    # 제외 대상(우선주)
    assert len(events.collect_events({"000815": df}, exclude={"000815"})) == 0
    # 마스터에 없는 신규상장/상장폐지 코드는 생존편향을 줄이므로 남긴다
    assert len(events.collect_events({"999999": df}, exclude={"000815"})) == 1
    # exclude=None 이면 필터를 걸지 않는다
    assert len(events.collect_events({"000815": df}, exclude=None)) == 1


def test_add_market_adjusted_그날_중앙값을_뺀다():
    ev = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"]),
        "y_raw": [0.01, 0.02, 0.03, 0.05],
    })
    out = events.add_market_adjusted(ev)
    # 1/2 중앙값 0.02 → -0.01, 0.0, 0.01 / 1/3은 단일이라 0.0
    assert list(out["y_mkt"].round(6)) == [-0.01, 0.0, 0.01, 0.0]
