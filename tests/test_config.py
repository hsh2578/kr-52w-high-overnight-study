# -*- coding: utf-8 -*-
from pathlib import Path

from surge_d1 import config


def test_상수값이_스펙과_일치한다():
    assert config.COST == 0.0021
    assert config.HIGH_52W_PERIOD == 252
    assert config.MIN_TRADING_VALUE == 1e10
    assert config.VOLUME_SURGE_MULT == 1.5
    assert config.MIN_BAR_INDEX == 260
    assert config.HALT_LOOKBACK == 5
    assert config.MAX_DAILY_MOVE == 0.32
    assert config.EMBARGO_DAYS == 5
    assert config.RANDOM_SEED == 42


def test_경로가_Path객체다():
    for name in ("PROJECT_ROOT", "HOLLYKR_CACHE", "CACHE_PATH", "CATALYST_CSV"):
        assert isinstance(getattr(config, name), Path), name


def test_원본캐시경로는_프로젝트_바깥이다():
    # 원본은 읽기 전용이므로 프로젝트 안에 있으면 안 된다
    assert config.PROJECT_ROOT not in config.HOLLYKR_CACHE.parents


def test_갱신본_캐시는_프로젝트_안이다():
    assert config.PROJECT_ROOT in config.CACHE_PATH.parents
