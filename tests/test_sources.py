# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from surge_d1 import sources


def _df(dates, o, h, l, c, v):
    return pd.DataFrame(
        {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v},
        index=pd.to_datetime(dates),
    )


def test_build_market_index_등가중수익률과_갭을_계산한다():
    d = ["2024-01-02", "2024-01-03"]
    ohlcv = {
        "000001": _df(d, [100, 110], [110, 120], [95, 105], [100, 120], [10, 10]),
        "000002": _df(d, [200, 210], [210, 220], [195, 205], [200, 220], [10, 10]),
    }
    mi = sources.build_market_index(ohlcv)
    # 첫날은 전일이 없어 NaN
    assert np.isnan(mi.loc[pd.Timestamp("2024-01-02"), "ret"])
    # 둘째날 수익률: (120-100)/100=0.2, (220-200)/200=0.1 → 평균 0.15
    assert mi.loc[pd.Timestamp("2024-01-03"), "ret"] == pytest.approx(0.15)
    # 둘째날 갭: (110-100)/100=0.1, (210-200)/200=0.05 → 평균 0.075
    assert mi.loc[pd.Timestamp("2024-01-03"), "gap"] == pytest.approx(0.075)
    assert mi.loc[pd.Timestamp("2024-01-03"), "n"] == 2


def test_build_market_index_종목마다_거래일이_달라도_처리한다():
    ohlcv = {
        "000001": _df(["2024-01-02", "2024-01-03"], [100, 110], [110, 120],
                      [95, 105], [100, 120], [10, 10]),
        "000002": _df(["2024-01-03"], [200], [210], [195], [220], [10]),
    }
    mi = sources.build_market_index(ohlcv)
    # 000002는 전일이 없어 둘째날 수익률 계산 불가 → 000001만 반영
    assert mi.loc[pd.Timestamp("2024-01-03"), "ret"] == pytest.approx(0.2)
    assert mi.loc[pd.Timestamp("2024-01-03"), "n"] == 1


def test_mark_halted_거래정지봉을_표시하고_NaN으로_바꾼다():
    # 원본은 거래정지일을 Open=0, Volume=0, Close=직전값으로 채운다
    d = ["2021-04-09", "2021-04-12", "2021-04-13", "2021-04-15"]
    df = _df(d, [111, 0, 0, 120], [112, 112, 112, 121],
             [110, 112, 112, 119], [112, 112, 112, 120], [100, 0, 0, 500])
    out = sources.mark_halted(df)
    assert list(out["halted"]) == [False, True, True, False]
    assert np.isnan(out["Open"].iloc[1])
    assert np.isnan(out["Volume"].iloc[2])
    # 정상 봉은 그대로
    assert out["Open"].iloc[0] == 111
    assert out["Close"].iloc[1] == 112       # 종가는 건드리지 않는다


def test_mark_halted_적용후_거래량평균이_정지봉을_건너뛴다():
    d = ["2021-04-09", "2021-04-12", "2021-04-15"]
    df = _df(d, [100, 0, 100], [101, 101, 101], [99, 101, 99],
             [100, 100, 100], [200, 0, 400])
    out = sources.mark_halted(df)
    assert out["Volume"].mean() == pytest.approx(300.0)   # 0을 빼고 (200+400)/2


def test_load_catalyst_컬럼과_타입을_정규화한다(tmp_path, monkeypatch):
    csv = tmp_path / "cat.csv"
    csv.write_text(
        "날짜,종목명,코드,당일등락률,당일등락률pct,재료유형,사유,익일D1,D2,D3,3일누적\n"
        "2025-07-01,테스트,65650,+29.29%,29.29,테마·수혜,대규모 수주,-18.06,-12.93,-5.64,-32.68\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sources.config, "CATALYST_CSV", csv)
    df = sources.load_catalyst()
    assert list(df.columns) == ["date", "code", "ctype", "reason", "day_chg", "d1"]
    assert df.loc[0, "code"] == "065650"          # 6자리 0패딩
    assert df.loc[0, "date"] == pd.Timestamp("2025-07-01")
    assert df.loc[0, "d1"] == pytest.approx(-18.06)
    assert df.loc[0, "reason"] == "대규모 수주"
