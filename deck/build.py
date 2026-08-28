# -*- coding: utf-8 -*-
"""슬라이드 조각을 하나의 deck-stage 덱으로 합친다.

조각을 나눈 이유: 14장을 한 파일에 두면 한 장을 고칠 때마다 전체를 다시 써야
해서 실수가 늘어난다. 11~13장은 16:9 판면을 넘겨 다시 짠 판이 따로 있으므로,
슬라이드 id를 명시해 순서대로 합친다 — id를 명시하지 않으면 slides-8-14.html에
남아 있는 옛 11~13장이 조용히 끌려와 고친 판을 덮는다.

사용: python build.py
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")   # 콘솔 기본이 cp949라 한글/기호가 깨진다

HERE = Path(__file__).resolve().parent
OUT = HERE / "발표.html"

W, H = 1600, 900

# 순서가 곧 발표 순서다. (파일, 그 파일에서 가져올 슬라이드 id)
PARTS = [
    ("slides-1-7.html",   ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]),
    ("slides-8-14.html",  ["s8", "s9", "s10"]),
    ("slides-11-13.html", ["s11", "s12"]),
    ("slides-limitup.html", ["s12b"]),
    ("slides-11-13.html", ["s13"]),
    ("slides-8-14.html",  ["s14"]),
]

# 발표자 도구·댓글에서 장을 부를 이름. 순서는 PARTS를 펼친 순서와 같아야 한다.
LABELS = [
    "표지", "목차", "왜 이 연구를 시작했나", "예측 지평을 하루로 고정한 이유",
    "무엇을 돌파로 볼 것인가", "유니버스 — 보통주만", "데이터 수집과 정제",
    "변수 설계 55개", "모델과 검증 설계", "결과 ① 이벤트의 알파",
    "결과 ② 확률 보정 버그", "결과 ③ 그냥 돌파 vs 선별 시스템",
    "결과 ④ 그 픽의 74%는 살 수 없었다",
    "결과 ⑤ 사지 않는 날의 값어치", "한계와 결론",
]

HEAD = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>52주 신고가 돌파 종목의 익일 초과수익 — FIND-A TEAM Beta</title>
<link rel="stylesheet" href="style.css?v=__CSSV__">
<script src="deck-stage.js"></script>
</head>
<body>
<deck-stage width="__W__" height="__H__">
"""

TAIL = """</deck-stage>
</body>
</html>
"""


def extract(src: str, sid: str) -> str:
    """<section class="slide ..." id="sid"> ... </section> 한 덩어리를 떼어온다.

    클래스 목록을 정확히 맞추지 않는 이유: 판면이 빠듯한 슬라이드에는
    `slide tight`처럼 수정자가 붙는다. 문자열을 그대로 찾으면 수정자를
    추가하는 순간 빌드가 깨진다.
    """
    m = re.search(rf'<section class="slide[^"]*" id="{sid}">', src)
    if not m:
        raise KeyError(f"슬라이드 {sid}를 찾지 못했다")
    end = src.index("</section>", m.start()) + len("</section>")
    return src[m.start():end]


def label_slide(html: str, label: str) -> str:
    """deck-stage가 읽는 data-label과 에디터 댓글용 data-screen-label을 붙인다."""
    esc = label.replace('"', "&quot;")
    return re.sub(
        r'^<section ',
        f'<section data-label="{esc}" data-screen-label="{esc}" ',
        html, count=1)


def main() -> int:
    cache: dict[str, str] = {}
    ids = [sid for _, group in PARTS for sid in group]
    if len(ids) != len(LABELS):
        raise SystemExit(f"슬라이드 {len(ids)}개인데 이름은 {len(LABELS)}개다")

    chunks = []
    k = 0
    for fname, group in PARTS:
        if fname not in cache:
            cache[fname] = io.open(HERE / fname, encoding="utf-8").read()
        for sid in group:
            chunks.append(label_slide(extract(cache[fname], sid), LABELS[k]))
            k += 1

    # CSS에 버전 쿼리를 붙인다. 붙이지 않으면 스타일을 고쳐도 브라우저가 캐시된
    # 옛 파일을 계속 써서 "고쳤는데 안 바뀐다"로 시간을 버린다.
    cssv = str(int((HERE / "style.css").stat().st_mtime))
    head = HEAD.replace("__CSSV__", cssv).replace("__W__", str(W)).replace("__H__", str(H))
    out = head + "\n\n".join(chunks) + TAIL
    io.open(OUT, "w", encoding="utf-8").write(out)

    n = len(re.findall(r"<section ", out))
    print(f"{OUT.name} 생성 — 슬라이드 {n}장 / {len(out.encode()) // 1024} KB")
    assert n == 15, f"슬라이드가 15장이 아니다: {n}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
