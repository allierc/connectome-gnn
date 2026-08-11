"""Minimal Markdown -> LaTeX converter tailored to notebook.md (no pandoc).
Handles: YAML front-matter (title/author/date), ATX headers, fenced code
blocks (verbatim), pipe tables, bold **x**, italic *x*, inline `code`,
[text](url) links (url dropped, text kept), > blockquotes, and paragraphs.
Produces notebook.tex, then compile with pdflatex.

Defaults to notebook.md -> notebook.tex; pass --src/--out to convert any other
markdown doc in this folder, e.g.

    python docs/md2tex_notebook.py --src docs/HOWTO_add_oculomotor_circuit.md \\
        --out docs/HOWTO_add_oculomotor_circuit.tex --title "..."
"""
import argparse, re, sys, os

SRC = os.path.join(os.path.dirname(__file__), "notebook.md")
OUT = os.path.join(os.path.dirname(__file__), "notebook.tex")

# Non-ASCII that T1/pdflatex will not render as-is. Applied to prose only;
# code blocks go through the `literate` map in the lstset preamble instead.
UNICODE = {
    "—": "---", "–": "--", "§": r"\S{}",
    "→": r"$\rightarrow$", "π": r"$\pi$",
    "≈": r"$\approx$", "×": r"$\times$", "±": r"$\pm$",
    "≤": r"$\leq$", "≥": r"$\geq$", "ρ": r"$\rho$",
    "θ": r"$\theta$", "σ": r"$\sigma$", "λ": r"$\lambda$",
    "τ": r"$\tau$", "ω": r"$\omega$", "Δ": r"$\Delta$",
}


def esc(s):
    # escape LaTeX specials in prose (not in verbatim)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    for a, b in UNICODE.items():
        s = s.replace(a, b)
    return s


def inline(s):
    # protect inline math $...$ and code `...` from escaping
    store = []
    def stash(m):
        store.append(m.group(0))
        return f"\x00{len(store)-1}\x00"
    s = re.sub(r"\$[^$]*\$", stash, s)        # math passes through verbatim
    codes = []
    def stashcode(m):
        codes.append(m.group(1))
        return f"\x01{len(codes)-1}\x01"
    s = re.sub(r"`([^`]*)`", stashcode, s)
    # [text](url) -> text. Runs BEFORE esc() so the url's _ / # / % never reach
    # the escaper. The link text is usually the path already (e.g. a filename
    # in backticks), so dropping the target loses nothing on paper.
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", s)
    s = re.sub(r"\[\[(.+?)\]\]", r"\\textsf{[\1]}", s)
    # restore code (escape its contents minimally for texttt)
    for i, c in enumerate(codes):
        cc = c.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("#", r"\#").replace("&", r"\&")
        s = s.replace(f"\x01{i}\x01", f"\\texttt{{{cc}}}")
    for i, m in enumerate(store):
        s = s.replace(f"\x00{i}\x00", m)
    return s


def parse(lines):
    out, i, n = [], 0, len(lines)
    # front-matter
    title = author = date = None
    if lines and lines[0].strip() == "---":
        j = 1
        while j < n and lines[j].strip() != "---":
            m = re.match(r'(\w+):\s*"?(.*?)"?\s*$', lines[j])
            if m:
                k, v = m.group(1), m.group(2)
                if k == "title": title = v
                elif k == "author": author = v
                elif k == "date": date = v
            j += 1
        i = j + 1
    while i < n:
        ln = lines[i].rstrip("\n")
        # code fence
        if ln.strip().startswith("```"):
            buf = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i].rstrip("\n")); i += 1
            i += 1
            out.append("\\begin{lstlisting}")
            out.extend(buf)
            out.append("\\end{lstlisting}")
            continue
        # figure: a line that is nothing but ![caption](path). Must be tested
        # before the link rule in inline(), which would otherwise strip the
        # path and leave "!caption" as prose.
        m = re.match(r"!\[(.*)\]\((.*?)\)\s*$", ln.strip())
        if m:
            cap, src = m.group(1), m.group(2)
            out.append(r"\begin{figure}[H]\centering")
            out.append(r"\includegraphics[width=\textwidth]{%s}" % src)
            if cap.strip():
                out.append(r"\caption*{\small %s}" % inline(cap))
            out.append(r"\end{figure}")
            i += 1
            continue
        # headers
        m = re.match(r"(#{1,4})\s+(.*)", ln)
        if m:
            lvl = len(m.group(1)); txt = inline(m.group(2))
            cmd = {1: "section*", 2: "subsection*", 3: "subsubsection*",
                   4: "paragraph"}[lvl]
            out.append(f"\\{cmd}{{{txt}}}")
            i += 1
            continue
        # table block
        if ln.lstrip().startswith("|") and i + 1 < n and re.match(r"\s*\|?[\s:|-]+\|", lines[i+1]):
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                rows.append(lines[i].strip()); i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            header = cells[0]; body = cells[2:]
            ncol = len(header)
            # An l-column tabular cannot wrap, so a table with long cells runs
            # off the page. Switch to tabularx (X = wrapped, width-sharing)
            # once the widest row would not plausibly fit on one line.
            widest = max((sum(len(c) for c in r) for r in cells), default=0)
            wrap = widest > 80
            env = "tabularx" if wrap else "tabular"
            spec = ("{\\textwidth}{" + ">{\\raggedright\\arraybackslash}X" * ncol
                    if wrap else "{" + "l" * ncol)
            out.append("\\begin{center}\\small")
            out.append(f"\\begin{{{env}}}" + spec + "}\\toprule")
            out.append(" & ".join(inline(c) for c in header) + " \\\\\\midrule")
            for r in body:
                r = (r + [""] * ncol)[:ncol]
                out.append(" & ".join(inline(c) for c in r) + " \\\\")
            out.append(f"\\bottomrule\\end{{{env}}}\\end{{center}}")
            continue
        # blockquote
        if ln.lstrip().startswith(">"):
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(lines[i].lstrip()[1:].strip()); i += 1
            out.append("\\begin{quote}\\itshape " + inline(" ".join(buf)) + "\\end{quote}")
            continue
        # horizontal rule
        if ln.strip() == "---":
            out.append("\\par\\noindent\\rule{\\textwidth}{0.4pt}\\par")
            i += 1
            continue
        # list item
        if re.match(r"\s*[-*]\s+", ln):
            items = []
            while i < n and re.match(r"\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"\s*[-*]\s+", "", lines[i].rstrip("\n"), count=1))
                i += 1
            out.append("\\begin{itemize}\\setlength{\\itemsep}{1pt}")
            for it in items:
                out.append("  \\item " + inline(it))
            out.append("\\end{itemize}")
            continue
        # numbered list
        if re.match(r"\s*\d+\.\s+", ln):
            items = []
            while i < n and re.match(r"\s*\d+\.\s+", lines[i]):
                items.append(re.sub(r"\s*\d+\.\s+", "", lines[i].rstrip("\n"), count=1))
                i += 1
            out.append("\\begin{enumerate}\\setlength{\\itemsep}{1pt}")
            for it in items:
                out.append("  \\item " + inline(it))
            out.append("\\end{enumerate}")
            continue
        # blank
        if ln.strip() == "":
            out.append("")
            i += 1
            continue
        # paragraph (gather until blank)
        buf = [ln]
        i += 1
        while i < n and lines[i].strip() != "" and not re.match(r"(#{1,4}\s|```|\s*[-*]\s|\s*\d+\.\s|>|\|)", lines[i]) and lines[i].strip() != "---":
            buf.append(lines[i].rstrip("\n")); i += 1
        out.append(inline(" ".join(buf)))
    return title, author, date, out


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=2.3cm]{geometry}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{amssymb,amsmath}
\usepackage{parskip}
%% Long file paths in \texttt{} are single unbreakable words and overflow the
%% margin; [htt] lets them hyphenate like prose. (%% is doubled because this
%% preamble is a Python %%-format template.)
\usepackage[htt]{hyphenat}
\sloppy
\emergencystretch=2em
\definecolor{nbbg}{HTML}{f5f5f5}
\lstset{basicstyle=\ttfamily\footnotesize,backgroundcolor=\color{nbbg},
  breaklines=true,frame=single,framesep=4pt,columns=fullflexible,
  literate={±}{{$\pm$}}1 {≈}{{$\approx$}}1 {→}{{$\rightarrow$}}1 {×}{{$\times$}}1
           {λ}{{$\lambda$}}1 {ρ}{{$\rho$}}1 {θ}{{$\theta$}}1 {σ}{{$\sigma$}}1
           {²}{{\textsuperscript{2}}}1 {α}{{$\alpha$}}1 {<->}{{$\leftrightarrow$}}1
           {—}{{---}}1 {–}{{--}}1 {π}{{$\pi$}}1 {§}{{\S}}1}
\setcounter{secnumdepth}{0}
\title{%s}
\author{%s}
\date{%s}
\begin{document}
\maketitle
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=None,
                    help="output .tex (default: --src with a .tex suffix)")
    ap.add_argument("--title", default=None,
                    help="overrides the front-matter title")
    ap.add_argument("--author", default=None)
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    out = args.out or os.path.splitext(args.src)[0] + ".tex"
    with open(args.src) as f:
        lines = f.readlines()
    title, author, date, body = parse(lines)
    title = args.title or title or "Research notebook"
    author = args.author or author or ""
    date = args.date or date or ""
    with open(out, "w") as f:
        f.write(PREAMBLE % (title, author, date))
        f.write("\n".join(body))
        f.write("\n\\end{document}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
