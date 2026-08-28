/* deck-stage — 고정 판면 슬라이드를 어떤 화면에도 맞추는 무대.
 *
 * 계약:
 *   <deck-stage width="1600" height="900">
 *     <section data-label="표지"> ... </section>
 *     ...
 *   </deck-stage>
 *
 * 하는 일
 *   - 레터박스 스케일링: 슬라이드는 항상 width×height로 그려지고 transform:scale로만
 *     맞춘다. 반응형 재배치를 하지 않는 이유는, 발표 자료의 판면은 설계된 그대로
 *     보여야 하기 때문이다 — 화면이 좁다고 표가 접히면 발표가 망가진다.
 *   - 키보드(←/→/Space/Home/End)·클릭·스와이프 이동
 *   - 현재 장을 URL 해시(#3)에 유지해 새로고침·링크 공유에도 위치가 남는다
 *   - 인쇄 시 한 장당 한 페이지 (PDF 내보내기)
 *
 * 조작부는 스케일되는 요소 **바깥**에 둔다. 안에 두면 축소 배율만큼 같이 작아져
 * 작은 화면일수록 누르기 어려워진다.
 */
(function () {
  "use strict";

  var CSS = `
deck-stage{position:fixed;inset:0;display:block;background:#050505;overflow:hidden}
deck-stage > section{
  position:absolute;left:0;top:0;transform-origin:top left;
  display:none;overflow:hidden;
}
deck-stage > section[data-active]{display:block}
.ds-bar{
  position:fixed;right:20px;bottom:18px;z-index:9;
  display:flex;align-items:center;gap:6px;
  font:500 13px/1 'Noto Sans KR',system-ui,sans-serif;color:#8A8A8A;
  background:rgba(20,20,20,.82);border:1px solid #2A2A2A;border-radius:999px;
  padding:6px 8px;backdrop-filter:blur(6px);opacity:.25;transition:opacity .18s;
}
.ds-bar:hover{opacity:1}
.ds-bar button{
  all:unset;cursor:pointer;padding:5px 10px;border-radius:999px;
  color:#CFCFCF;font:inherit;
}
.ds-bar button:hover{background:#262626;color:#fff}
.ds-bar button[disabled]{opacity:.3;cursor:default}
.ds-bar .ds-pos{padding:0 8px;font-variant-numeric:tabular-nums;letter-spacing:.02em}

@media print{
  @page{size:landscape;margin:0}
  html,body{background:#fff !important}
  deck-stage{position:static;overflow:visible;background:none}
  deck-stage > section{
    position:relative !important;display:block !important;
    transform:none !important;break-after:page;page-break-after:always;
  }
  deck-stage > section:last-child{break-after:auto;page-break-after:auto}
  .ds-bar{display:none !important}
}
`;

  function injectCss() {
    if (document.getElementById("ds-style")) return;
    var s = document.createElement("style");
    s.id = "ds-style";
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  class DeckStage extends HTMLElement {
    connectedCallback() {
      if (this._ready) return;

      // connectedCallback은 여는 태그가 파싱되는 순간 불린다 — 이 스크립트가
      // <head>에 있으면 자식 <section>이 아직 하나도 없다. 실제로 그래서
      // 슬라이드를 0개로 읽고 화면이 비었다. 문서 파싱이 끝난 뒤에 시작한다.
      if (document.readyState === "loading") {
        var self = this;
        document.addEventListener("DOMContentLoaded", function () {
          self._init();
        }, { once: true });
        return;
      }
      this._init();
    }

    _init() {
      if (this._ready) return;
      this._ready = true;
      injectCss();

      this.W = +this.getAttribute("width") || 1600;
      this.H = +this.getAttribute("height") || 900;
      this.slides = Array.from(this.children).filter(function (el) {
        return el.tagName === "SECTION";
      });
      this.slides.forEach(function (el, i) {
        el.style.width = this.W + "px";
        el.style.height = this.H + "px";
        el.setAttribute("data-index", i + 1);
      }, this);

      this._buildBar();
      this._bind();
      this.go(this._fromHash(), true);
      this._fit();
    }

    _fromHash() {
      var n = parseInt((location.hash || "").replace("#", ""), 10);
      return isFinite(n) ? n : 1;
    }

    _buildBar() {
      var bar = document.createElement("div");
      bar.className = "ds-bar";
      bar.innerHTML =
        '<button data-ds="prev" aria-label="이전 슬라이드">←</button>' +
        '<span class="ds-pos"></span>' +
        '<button data-ds="next" aria-label="다음 슬라이드">→</button>' +
        '<button data-ds="print" aria-label="PDF로 인쇄">PDF</button>';
      document.body.appendChild(bar);
      this.bar = bar;

      var self = this;
      bar.addEventListener("click", function (e) {
        var b = e.target.closest("button");
        if (!b) return;
        var a = b.getAttribute("data-ds");
        if (a === "prev") self.go(self.i - 1);
        else if (a === "next") self.go(self.i + 1);
        else if (a === "print") window.print();
      });
    }

    _bind() {
      var self = this;
      addEventListener("resize", function () { self._fit(); });
      addEventListener("hashchange", function () { self.go(self._fromHash(), true); });

      addEventListener("keydown", function (e) {
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        var k = e.key;
        if (k === "ArrowRight" || k === "PageDown" || k === " ") { self.go(self.i + 1); e.preventDefault(); }
        else if (k === "ArrowLeft" || k === "PageUp") { self.go(self.i - 1); e.preventDefault(); }
        else if (k === "Home") { self.go(1); e.preventDefault(); }
        else if (k === "End") { self.go(self.slides.length); e.preventDefault(); }
      });

      // 클릭으로는 이동하지 않는다. 이 덱은 편집기 안에서도 열리는데, 거기서
      // 클릭은 '요소 선택'이다 — 클릭 이동을 붙였더니 요소를 고르려는 클릭마다
      // 슬라이드가 넘어가 편집이 불가능했다. 이동은 키보드·조작부·스와이프로만.
      var x0 = null;
      this.addEventListener("touchstart", function (e) { x0 = e.touches[0].clientX; }, { passive: true });
      this.addEventListener("touchend", function (e) {
        if (x0 === null) return;
        var dx = e.changedTouches[0].clientX - x0;
        if (Math.abs(dx) > 45) self.go(self.i + (dx < 0 ? 1 : -1));
        x0 = null;
      }, { passive: true });
    }

    _fit() {
      // 화면 안에 판면을 통째로 넣고 남는 여백은 위아래(또는 좌우)로 나눈다.
      var s = Math.min(innerWidth / this.W, innerHeight / this.H);
      var ox = (innerWidth - this.W * s) / 2;
      var oy = (innerHeight - this.H * s) / 2;
      this.slides.forEach(function (el) {
        el.style.transform = "translate(" + ox + "px," + oy + "px) scale(" + s + ")";
      });
    }

    go(n, silent) {
      var total = this.slides.length;
      n = Math.max(1, Math.min(total, n || 1));
      if (n === this.i) return;
      this.i = n;
      this.slides.forEach(function (el, k) {
        if (k === n - 1) el.setAttribute("data-active", "");
        else el.removeAttribute("data-active");
      });
      var pos = this.bar.querySelector(".ds-pos");
      if (pos) pos.textContent = n + " / " + total;
      this.bar.querySelector('[data-ds="prev"]').disabled = n === 1;
      this.bar.querySelector('[data-ds="next"]').disabled = n === total;
      if (!silent) history.replaceState(null, "", "#" + n);
      this._fit();
    }
  }

  if (!customElements.get("deck-stage")) {
    customElements.define("deck-stage", DeckStage);
  }
})();
