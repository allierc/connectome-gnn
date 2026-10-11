"""Copied from Plexus/tools/deck_site.py (2026-10-11) for the connectome-gnn paper site (tools/paper_site.py), with
two additions: web links in the PDF (\\href{https://...}) become clickable areas laid over the slide image, and the
number of sections the main page takes is a parameter (--min-sections/--max-sections; the paper has one per figure
and table).

A beamer deck as a static website (Cedric, 2026-10-09: "make a local website of exp17.pdf to share with my colleagues").

Every page of the compiled PDF becomes an image; every movie the deck plays (`\\playmovie`: a link `run:Movies/<x>.mp4`
over its poster frame) becomes an HTML5 video laid exactly over that frame, playing while its slide is shown. One
`index.html`, no server-side code, no external library: the folder opens from any web server or from a zip.

    python tools/deck_site.py experiments/exp17_zapbench_graphcast/presentation/exp17.pdf [--out <dir>] [--dpi 220]
    cd <dir> && python -m http.server 8860          # then http://127.0.0.1:8860/

-> <dir> (default: <pdf dir>/site): index.html, slides/NNN.png, Movies/<the movies the deck plays>.mp4
Keys: left / right (or space), Home / End; the slide is in the URL (#12), so a link points at one slide.

THE MAIN PAGE (Cedric, 2026-10-11: "a main page to jump to the different sections; at least five sections, at most
ten"): the site opens on a page of section cards (title, pages, the first slide as thumbnail); a card opens the viewer
at that slide; the house button at the slide's top right, "contents" in the sidebar, or Escape returns. The sections come from `--sections <json>`, an ordered
list of [name, regex]: a section starts at the first slide whose title line matches its regex and runs to the next
section's start; the default list is exp17's (DEFAULT_SECTIONS). The sidebar lists the slides under their sections.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import urllib.parse

DEFAULT_SECTIONS = [["the zebrafish data: ZAPBench, the atlas, the slow modes", r"^the zebrafish data"],   # exp17's (tools/exp17_b19_deck.py
                    ["the graphs", r"^the graphs"],                                   # SECTIONS: every slide title
                    ["the input neurons", r"^the input neurons"],                     # opens with its section's)
                    ["the model", r"^the model"],
                    ["the neuron graph results (batch 22)", r"^the neuron graph results"],
                    ["the lattice grid results (batch 23)", r"^the lattice grid results"],
                    ["the angle rule results (batch 24)", r"^the angle rule results"],
                    ["the input mask results (batch 25)", r"^the input mask results"],
                    ["the stimulus video results (batch 26)", r"^the stimulus video results"],
                    ["the alpha state-circuit results (batch 27, coming)", r"^the alpha state-circuit results"],
                    ["the appendix", r"^the appendix"]]

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #ddd; font: 14px/1.35 Helvetica, Arial, sans-serif; }
  #side { position: fixed; left: 0; top: 0; bottom: 0; width: 260px; overflow-y: auto; background: #1a1a1a;
          border-right: 1px solid #333; }
  #side h1 { font-size: 15px; margin: 12px 12px 6px; color: #fff; }
  #side .hint { margin: 0 12px 10px; color: #888; font-size: 12px; }
  #side a { display: block; padding: 4px 12px; color: #bbb; text-decoration: none; font-size: 12.5px; }
  #side a.on { background: #2e7d32; color: #fff; }
  #side a:hover { background: #333; }
  #main { position: fixed; left: 261px; right: 0; top: 0; bottom: 0; display: flex; align-items: center;
          justify-content: center; }
  #stage { position: relative; }
  #stage img { display: block; width: 100%; height: 100%; }
  #stage video { position: absolute; background: #000; }
  #stage a.lnk { position: absolute; display: block; }
  #stage a.lnk:hover { outline: 1px solid #8fd18f; background: rgba(143, 209, 143, 0.18); }
  #nav { position: fixed; right: 14px; bottom: 10px; color: #888; font-size: 12px; }
  #homebtn { position: fixed; right: 14px; top: 48px; width: 34px; height: 34px; padding: 7px; background: #2e7d32;
             border: 0; border-radius: 4px; cursor: pointer; }   /* flat: one colour, no shadow (Cedric, 2026-10-11) */
  #homebtn svg { display: block; width: 20px; height: 20px; fill: #fff; }
  #homebtn:hover { background: #3a9a40; }
  #nav button { background: #333; color: #ddd; border: 0; padding: 4px 10px; margin-left: 4px; cursor: pointer; }
  #side .sec { display: block; padding: 8px 12px 3px; color: #8fd18f; font-size: 11px; text-transform: uppercase;
               letter-spacing: 0.04em; }
  #side .home { display: block; padding: 6px 12px; color: #fff; background: #244; text-decoration: none; font-size: 12.5px; }
  #home { position: fixed; left: 261px; right: 0; top: 0; bottom: 0; overflow-y: auto; padding: 28px 36px; }
  #home h2 { margin: 0 0 4px; font-size: 22px; color: #fff; }
  #home .sub { color: #999; margin-bottom: 22px; }
  #cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 18px; }
  .card { background: #1c1c1c; border: 1px solid #333; cursor: pointer; }
  .card:hover { border-color: #2e7d32; }
  .card img { display: block; width: 100%; }
  .card .t { padding: 10px 12px 2px; color: #fff; font-size: 15px; }
  .card .p { padding: 0 12px 10px; color: #999; font-size: 12px; }
  @media (max-width: 800px) { #side { display: none; } #main, #home { left: 0; } }
</style></head>
<body>
<div id="side"><h1>__TITLE__</h1><div class="hint">__HINT__</div><a class="home" href="#home" id="homelink">contents</a><div id="toc"></div></div>
<div id="home"><h2>__TITLE__</h2><div class="sub">__HINT__ &middot; a card opens its first slide; Escape returns here</div><div id="cards"></div></div>
<div id="main"><div id="stage"></div><button id="homebtn" title="contents (Escape)" aria-label="contents"><svg viewBox="0 0 24 24"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg></button></div>
<div id="nav"><span id="count"></span><button id="prev">&larr;</button><button id="next">&rarr;</button></div>
<script>
const S = __SLIDES__;
const SEC = __SECTIONS__;
const AR = __AR__;
let cur = 0;
const stage = document.getElementById("stage"), toc = document.getElementById("toc");
const home = document.getElementById("home"), main = document.getElementById("main"), cards = document.getElementById("cards");
S.forEach((s, i) => {
  const sec = SEC.find((c) => c.start === i);
  if (sec) { const h = document.createElement("span"); h.className = "sec"; h.textContent = sec.name; toc.appendChild(h); }
  const a = document.createElement("a"); a.href = "#" + (i + 1); a.textContent = (i + 1) + "  " + s.title;
  a.onclick = (e) => { e.preventDefault(); show(i); }; toc.appendChild(a);
});
SEC.forEach((c) => {
  const d = document.createElement("div"); d.className = "card";
  d.innerHTML = "<img src=\"" + S[c.start].img + "\" alt=\"\"><div class=\"t\">" + c.name + "</div><div class=\"p\">slides "
    + (c.start + 1) + " to " + (c.end + 1) + "</div>";
  d.onclick = () => show(c.start); cards.appendChild(d);
});
function goHome() {
  main.style.display = "none"; home.style.display = "block"; history.replaceState(null, "", "#home");
  [...toc.children].forEach((a) => a.classList && a.classList.remove("on"));
}
document.getElementById("homelink").onclick = (e) => { e.preventDefault(); goHome(); };
document.getElementById("homebtn").onclick = goHome;
function fit() {
  const m = document.getElementById("main"), W = m.clientWidth - 20, H = m.clientHeight - 40;
  const w = Math.min(W, H * AR); stage.style.width = w + "px"; stage.style.height = (w / AR) + "px";
}
function show(i) {
  home.style.display = "none"; main.style.display = "flex"; fit();
  cur = Math.max(0, Math.min(S.length - 1, i));
  const s = S[cur];
  stage.innerHTML = "";
  const img = document.createElement("img"); img.src = s.img; img.alt = s.title; stage.appendChild(img);
  s.movies.forEach((m) => {
    const v = document.createElement("video");
    v.src = m.src; v.muted = true; v.loop = true; v.autoplay = true; v.playsInline = true; v.controls = true;
    v.style.left = m.x + "%"; v.style.top = m.y + "%"; v.style.width = m.w + "%"; v.style.height = m.h + "%";
    stage.appendChild(v);
  });
  (s.links || []).forEach((l) => {
    const a = document.createElement("a"); a.className = "lnk"; a.href = l.href; a.target = "_blank"; a.title = l.href;
    a.style.left = l.x + "%"; a.style.top = l.y + "%"; a.style.width = l.w + "%"; a.style.height = l.h + "%";
    stage.appendChild(a);
  });
  const links = [...toc.querySelectorAll("a")];
  links.forEach((a, k) => a.classList.toggle("on", k === cur));
  links[cur].scrollIntoView({block: "nearest"});
  document.getElementById("count").textContent = (cur + 1) + " / " + S.length;
  history.replaceState(null, "", "#" + (cur + 1));
  [cur + 1, cur - 1].forEach((k) => { if (S[k]) { const p = new Image(); p.src = S[k].img; } });
}
document.addEventListener("keydown", (e) => {
  if (["ArrowRight", "PageDown", " "].includes(e.key)) { e.preventDefault(); show(cur + 1); }
  if (["ArrowLeft", "PageUp"].includes(e.key)) { e.preventDefault(); show(cur - 1); }
  if (e.key === "Home") show(0);
  if (e.key === "End") show(S.length - 1);
  if (e.key === "Escape") goHome();
});
window.addEventListener("hashchange", () => { const h = location.hash.slice(1); if (h === "home" || !h) goHome(); else show(parseInt(h) - 1); });
document.getElementById("prev").onclick = () => show(cur - 1);
document.getElementById("next").onclick = () => show(cur + 1);
window.addEventListener("resize", fit);
fit(); const h0 = location.hash.slice(1); if (!h0 || h0 === "home") goHome(); else show(parseInt(h0) - 1);
</script></body></html>
"""


def sections(slides, rules):
    """[{name, start, end}] from the slides' title lines and the ordered [name, regex] rules: a section starts at the
    first slide whose title matches its rule (searched after the previous section's start) and ends before the next."""
    import re
    out, pos = [], 0
    for name, rx in rules:
        hit = next((i for i in range(pos, len(slides)) if re.search(rx, slides[i]["title"])), None)
        if hit is None:
            continue
        out.append({"name": name, "start": hit})
        pos = hit + 1
    for k, c in enumerate(out):
        c["end"] = (out[k + 1]["start"] - 1) if k + 1 < len(out) else len(slides) - 1
    return out


def build(pdf, out=None, dpi=220, title=None, rules=None, min_sections=5, max_sections=12, first_title=None):
    import fitz
    base = os.path.dirname(os.path.abspath(pdf))
    out = out or os.path.join(base, "site")
    os.makedirs(os.path.join(out, "slides"), exist_ok=True)
    os.makedirs(os.path.join(out, "Movies"), exist_ok=True)
    d = fitz.open(pdf)
    W, H = d[0].rect.width, d[0].rect.height
    slides, movies = [], set()
    stamp = int(os.path.getmtime(pdf))
    for p in range(d.page_count):
        pg = d[p]
        img = f"slides/{p + 1:03d}.png"
        pg.get_pixmap(dpi=dpi).save(os.path.join(out, img))
        lines = [t.strip() for t in pg.get_text().split("\n") if t.strip()]
        mv, lk = [], []
        for ln in pg.get_links():
            uri = ln.get("uri") or ""
            if uri.startswith(("http://", "https://")):
                r = ln["from"]
                lk.append({"href": uri, "x": 100 * r.x0 / W, "y": 100 * r.y0 / H,
                           "w": 100 * r.width / W, "h": 100 * r.height / H})
                continue
            f = urllib.parse.unquote(ln.get("file") or ln.get("uri") or "").replace("run:", "").split("?")[0]
            if not f.endswith(".mp4"):
                continue
            src = os.path.join(base, f)
            if not os.path.exists(src):
                continue
            movies.add(f)
            r = ln["from"]
            mv.append({"src": f, "x": 100 * r.x0 / W, "y": 100 * r.y0 / H, "w": 100 * r.width / W, "h": 100 * r.height / H})
        # ?v=<build time>: a browser that cached an older build's slide images loads the new ones (Cedric, 2026-10-10:
        # "I do not see the modifications" -- the page showed a cached 61-page build after the 65-page one was served)
        # the title page has no frame title: its first text line is a name, so the menu takes `first_title`
        name_p = first_title if (p == 0 and first_title) else (lines[0] if lines else f"slide {p + 1}")
        slides.append({"img": f"{img}?v={stamp}", "title": name_p, "movies": mv,
                       "links": lk})
    for f in sorted(movies):
        dst = os.path.join(out, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(os.path.join(base, f)):
            shutil.copy2(os.path.join(base, f), dst)
    name = title or os.path.splitext(os.path.basename(pdf))[0]
    secs = sections(slides, rules if rules is not None else DEFAULT_SECTIONS)
    if not min_sections <= len(secs) <= max_sections:   # exp17: 5 to 12 (Cedric, 2026-10-11)
        raise SystemExit(f"[deck_site] {len(secs)} sections found; the main page takes {min_sections} to {max_sections} "
                         f"(rules: {len(rules or DEFAULT_SECTIONS)})")
    page = (PAGE.replace("__TITLE__", html.escape(name)).replace("__SLIDES__", json.dumps(slides))
            .replace("__SECTIONS__", json.dumps(secs))
            .replace("__AR__", f"{W / H:.6f}")
            .replace("__HINT__", f"{len(slides)} slides, {len(movies)} movies &middot; &larr; &rarr; to move"))
    open(os.path.join(out, "index.html"), "w").write(page)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)
    print(f"[deck_site] {len(slides)} slides, {len(movies)} movies, {len(secs)} sections, {size / 1e6:.0f} MB -> {out}/index.html")
    for c in secs:
        print(f"   {c['start'] + 1:3d}-{c['end'] + 1:<3d} {c['name']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--title", default=None)
    ap.add_argument("--sections", default=None, help="json: ordered [[name, regex], ...]; default exp17's")
    a = ap.parse_args()
    build(a.pdf, a.out, a.dpi, a.title, json.load(open(a.sections)) if a.sections else None)
