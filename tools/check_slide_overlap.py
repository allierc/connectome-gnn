"""Check a beamer deck (Janelia theme) for content overlapping the title bar or the footer.

    python tools/check_slide_overlap.py presentation/meeting_2026-10-12.pdf [more.pdf ...]

The theme paints the title bar as a full-width filled rectangle at the top and the footer as
text plus a progress bar at the bottom; neither is a frame element, so LaTeX does not stop
content from running over them. For every page but the title page: the title bar's bottom is
the lowest edge of the full-width filled rectangles touching y = 0, the footer's top is the
highest edge of the text and full-width rectangles in the bottom 12 % of the page. Any image,
text block or drawing of the slide body that crosses either line is reported. Exit code 1 if any.
"""
import sys

import fitz


def check(path):
    bad = 0
    doc = fitz.open(path)
    for k, pg in enumerate(doc):
        if k == 0:
            continue
        W, H = pg.rect.width, pg.rect.height
        draws = pg.get_drawings()
        bars = [d["rect"] for d in draws if d.get("fill") and d["rect"].y0 <= 0.5 and d["rect"].width > 0.9 * W
                and d["rect"].height < 0.25 * H]
        top = max((r.y1 for r in bars), default=0.0)
        blocks = pg.get_text("dict")["blocks"]
        foot = [b["bbox"][1] for b in blocks if b["type"] == 0 and b["bbox"][1] > 0.88 * H]
        foot += [d["rect"].y0 for d in draws if d.get("fill") and d["rect"].y0 > 0.88 * H and d["rect"].width > 0.05 * W]
        bottom = min(foot, default=H)
        hits = []
        for im in pg.get_image_info():
            y0, y1 = im["bbox"][1], im["bbox"][3]
            if y0 < top - 0.5 or y1 > bottom + 0.5:
                hits.append(f"image {y0:.0f}-{y1:.0f}")
        for b in blocks:
            if b["type"] != 0:
                continue
            y0, y1 = b["bbox"][1], b["bbox"][3]
            if y1 <= top + 1 or y0 >= bottom - 0.5:      # the title itself, the footer itself
                continue
            if y0 < top - 0.5 or y1 > bottom + 0.5:
                hits.append(f"text {y0:.0f}-{y1:.0f} '{''.join(s['text'] for l in b['lines'] for s in l['spans'])[:30]}'")
        if hits:
            bad += 1
            print(f"{path} slide {k + 1}: bar ends {top:.0f}, footer starts {bottom:.0f}: " + "; ".join(hits[:4]))
    print(f"{path}: {bad} of {len(doc) - 1} slides overlap the title bar or the footer")
    return bad


if __name__ == "__main__":
    sys.exit(1 if sum(check(p) for p in sys.argv[1:]) else 0)
