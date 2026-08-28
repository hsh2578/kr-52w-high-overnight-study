# -*- coding: utf-8 -*-
import pickle
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
    # 원본은 거래정지일을 Open=High=Low=0, Volume=0, Close=직전값으로 채운다
    # (예: 099520 2021-03-26)
    d = ["2021-04-09", "2021-04-12", "2021-04-13", "2021-04-15"]
    df = _df(d, [111, 0, 0, 120], [112, 0, 0, 121],
             [110, 0, 0, 119], [112, 112, 112, 120], [100, 0, 0, 500])
    out = sources.mark_halted(df)
    assert list(out["halted"]) == [False, True, True, False]
    assert np.isnan(out["Open"].iloc[1])
    assert np.isnan(out["High"].iloc[1])
    assert np.isnan(out["Low"].iloc[1])
    assert np.isnan(out["Volume"].iloc[2])
    # 정상 봉은 그대로
    assert out["Open"].iloc[0] == 111
    assert out["High"].iloc[0] == 112
    assert out["Low"].iloc[0] == 110
    assert out["Close"].iloc[1] == 112       # 종가는 건드리지 않는다


def test_mark_halted_전형적인_정지봉에서_High_Low도_NaN이_되고_Close는_유지된다():
    # 099520 2021-03-26 패턴: Open=High=Low=0, Volume=0, Close=직전값 이월
    d = ["2021-03-25", "2021-03-26", "2021-03-29"]
    df = _df(d, [1300, 0, 1290], [1310, 0, 1295],
             [1280, 0, 1280], [1284, 1284, 1284], [1000, 0, 800])
    out = sources.mark_halted(df)
    assert list(out["halted"]) == [False, True, False]
    assert np.isnan(out["Open"].iloc[1])
    assert np.isnan(out["High"].iloc[1])
    assert np.isnan(out["Low"].iloc[1])
    assert np.isnan(out["Volume"].iloc[1])
    assert out["Close"].iloc[1] == 1284       # 종가는 직전값 그대로 유지


def test_mark_halted_적용후_Low최솟값이_0이_아니다():
    # 정지봉의 Low=0이 그대로 남으면 후속 피처(RSV_20 등)의 min(Low)가
    # 0으로 붕괴한다 — 이를 방지하는 것이 이 수정의 핵심.
    d = ["2021-03-25", "2021-03-26", "2021-03-29"]
    df = _df(d, [1300, 0, 1290], [1310, 0, 1295],
             [1280, 0, 1280], [1284, 1284, 1284], [1000, 0, 800])
    out = sources.mark_halted(df)
    assert out["Low"].min() > 0
    assert out["High"].max() > 0


def test_mark_halted_High만_0이어도_정지로_잡힌다():
    # Open/Low/Volume은 정상인데 High만 0인 비정상 봉도 정지로 판정돼야 한다
    d = ["2021-03-25", "2021-03-26", "2021-03-29"]
    df = _df(d, [1300, 1290, 1290], [1310, 0, 1295],
             [1280, 1280, 1280], [1284, 1284, 1284], [1000, 900, 800])
    out = sources.mark_halted(df)
    assert list(out["halted"]) == [False, True, False]
    assert np.isnan(out["Open"].iloc[1])
    assert np.isnan(out["Low"].iloc[1])
    assert np.isnan(out["Volume"].iloc[1])


def test_mark_halted_불변식이_이미_성립하는_정상봉은_그대로_둔다():
    # High/Low가 이미 Open/Close를 포함하는 정상 봉은 값이 하나도 바뀌면 안 된다
    d = ["2021-03-25"]
    df = _df(d, [100], [105], [98], [102], [500])
    out = sources.mark_halted(df)
    assert out["halted"].iloc[0] == False
    assert out["Open"].iloc[0] == 100
    assert out["High"].iloc[0] == 105
    assert out["Low"].iloc[0] == 98
    assert out["Close"].iloc[0] == 102
    assert out["Volume"].iloc[0] == 500


def test_mark_halted_Close가_High보다_크면_High를_Close로_올린다():
    # 분할·액면가 조정 반올림 오차로 Close가 High를 넘어서는 봉(약 0.2%) —
    # High는 그날 거래된 모든 가격 이상이어야 한다는 정의를 어긴 것이므로 복구한다
    d = ["2021-03-25"]
    df = _df(d, [100], [105], [98], [108], [500])   # Close(108) > High(105)
    out = sources.mark_halted(df)
    assert out["halted"].iloc[0] == False
    assert out["High"].iloc[0] == 108
    assert out["Open"].iloc[0] == 100                # Open/Close는 원본 그대로 신뢰
    assert out["Close"].iloc[0] == 108
    assert out["Low"].iloc[0] == 98


def test_mark_halted_Low가_Open보다_크면_Low를_Open으로_내린다():
    d = ["2021-03-25"]
    df = _df(d, [100], [110], [102], [105], [500])  # Low(102) > Open(100)
    out = sources.mark_halted(df)
    assert out["halted"].iloc[0] == False
    assert out["Low"].iloc[0] == 100
    assert out["Open"].iloc[0] == 100
    assert out["Close"].iloc[0] == 105
    assert out["High"].iloc[0] == 110


def test_mark_halted_정지봉은_불변식_복구후에도_High_Low가_NaN으로_남는다():
    # 회귀 테스트 핵심: np.maximum/np.minimum의 NaN 전파가 깨지면
    # 정지봉의 NaN High/Low가 Close(1284)로 되살아난다 — 반드시 NaN이어야 한다
    d = ["2021-03-26"]
    df = _df(d, [0], [0], [0], [1284], [0])
    out = sources.mark_halted(df)
    assert out["halted"].iloc[0] == True
    assert np.isnan(out["High"].iloc[0])
    assert np.isnan(out["Low"].iloc[0])
    assert out["Close"].iloc[0] == 1284


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


def test_mark_halted_이미_NaN인_Open_Volume도_정지로_감지한다():
    # Open이 이미 NaN인 봉, Volume이 이미 NaN인 봉 모두 halted로 잡혀야 한다
    d = ["2021-04-09", "2021-04-12", "2021-04-13", "2021-04-14"]
    df = _df(
        d,
        [111, np.nan, 100, 105],   # Open: 정상, NaN, 정상, 정상
        [112, 112, 112, 112],
        [110, 112, 99, 104],
        [112, 112, 100, 105],
        [100, 50, np.nan, 300],    # Volume: 정상, 정상, NaN, 정상
    )
    out = sources.mark_halted(df)
    assert list(out["halted"]) == [False, True, True, False]


def test_load_ohlcv_갱신캐시가_있으면_그것을_그대로_쓴다(tmp_path, monkeypatch):
    d = ["2024-01-02", "2024-01-03"]
    df = _df(d, [100, 110], [110, 120], [95, 105], [100, 120], [10, 10])
    cache_path = tmp_path / "ohlcv_cache.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump({"000001": df}, f)
    monkeypatch.setattr(sources.config, "CACHE_PATH", cache_path)

    out = sources.load_ohlcv()

    assert list(out.keys()) == ["000001"]
    assert "halted" in out["000001"].columns
    assert isinstance(out["000001"].index, pd.DatetimeIndex)


def test_load_ohlcv_갱신캐시가_없으면_홀리크원본으로_대체한다(tmp_path, monkeypatch):
    d = ["2024-01-02", "2024-01-03"]
    df = _df(d, [100, 110], [110, 120], [95, 105], [100, 120], [10, 10])
    monkeypatch.setattr(sources.config, "CACHE_PATH", tmp_path / "missing" / "ohlcv_cache.pkl")

    hollykr_root = tmp_path / "hollykr"
    (hollykr_root / "ohlcv").mkdir(parents=True)
    with open(hollykr_root / "ohlcv" / "ohlcv_cache.pkl", "wb") as f:
        pickle.dump({"000001_260": df}, f)
    monkeypatch.setattr(sources.config, "HOLLYKR_CACHE", hollykr_root)

    out = sources.load_ohlcv()

    assert list(out.keys()) == ["000001"]
    assert "halted" in out["000001"].columns


def test_load_ohlcv_빈_갱신캐시는_원본으로_대체한다(tmp_path, monkeypatch):
    cache_path = tmp_path / "ohlcv_cache.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump({}, f)
    monkeypatch.setattr(sources.config, "CACHE_PATH", cache_path)

    d = ["2024-01-02"]
    df = _df(d, [100], [110], [95], [100], [10])
    hollykr_root = tmp_path / "hollykr"
    (hollykr_root / "ohlcv").mkdir(parents=True)
    with open(hollykr_root / "ohlcv" / "ohlcv_cache.pkl", "wb") as f:
        pickle.dump({"000001_260": df}, f)
    monkeypatch.setattr(sources.config, "HOLLYKR_CACHE", hollykr_root)

    out = sources.load_ohlcv()

    assert set(out.keys()) == {"000001"}


def test_load_ohlcv_같은코드_다른길이는_긴쪽이_이긴다(tmp_path, monkeypatch):
    short = _df(["2024-01-03"], [110], [120], [105], [120], [10])
    long = _df(["2024-01-02", "2024-01-03"], [100, 110], [110, 120],
               [95, 105], [100, 120], [10, 10])
    cache_path = tmp_path / "ohlcv_cache.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump({"000001_120": short, "000001_260": long}, f)
    monkeypatch.setattr(sources.config, "CACHE_PATH", cache_path)

    out = sources.load_ohlcv()

    assert list(out.keys()) == ["000001"]
    assert len(out["000001"]) == 2   # 더 긴(2일치) 쪽이 선택되어야 한다


def test_load_ohlcv_None이거나_빈_DataFrame은_건너뛴다(tmp_path, monkeypatch):
    df = _df(["2024-01-02"], [100], [110], [95], [100], [10])
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    cache_path = tmp_path / "ohlcv_cache.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump({"000001": df, "000002": None, "000003": empty}, f)
    monkeypatch.setattr(sources.config, "CACHE_PATH", cache_path)

    out = sources.load_ohlcv()

    assert set(out.keys()) == {"000001"}


def test_load_ohlcv_혼재된_키_형태를_올바르게_처리한다(tmp_path, monkeypatch):
    # 플레인 code key가 dict의 첫 항목이어도, 접미사가 붙은 다른 key는
    # 접미사가 벗겨진 code로 반환돼야 한다 (첫 key만 보고 전체 형태를
    # 판단하면 이 케이스에서 "000002_260"이 그대로 새어나간다).
    plain = _df(["2024-01-02"], [100], [110], [95], [100], [10])
    suffixed = _df(["2024-01-02", "2024-01-03"], [200, 210], [210, 220],
                    [195, 205], [200, 220], [10, 10])
    cache_path = tmp_path / "ohlcv_cache.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump({"000001": plain, "000002_260": suffixed}, f)
    monkeypatch.setattr(sources.config, "CACHE_PATH", cache_path)

    out = sources.load_ohlcv()

    assert set(out.keys()) == {"000001", "000002"}
