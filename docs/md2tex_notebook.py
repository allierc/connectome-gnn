"""Minimal Markdown -> LaTeX converter tailored to notebook.md (no pandoc).
Handles: YAML front-matter (title/author/date), ATX headers, fenced code
blocks (verbatim), pipe tables, bold **x**, italic *x*, inline `code`,
> blockquotes, and paragraphs. Produces notebook.tex, then compile with pdflatex.
"""
import re, sys, os

SRC = os.path.join(os.path.dirname(__file__), "notebook.md")
OUT = os.path.join(os.path.dirname(__file__), "notebook.tex")


def esc(s):
    # escape LaTeX specials in prose (not in verbatim)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
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
            out.append("\\begin{center}\\small")
            out.append("\\begin{tabular}{" + "l" * ncol + "}\\toprule")
            out.append(" & ".join(inline(c) for c in header) + " \\\\\\midrule")
            for r in body:
                r = (r + [""] * ncol)[:ncol]
                out.append(" & ".join(inline(c) for c in r) + " \\\\")
            out.append("\\bottomrule\\end{tabular}\\end{center}")
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
\usepackage{xcolor}
\usepackage{listings}
\usepackage{amssymb,amsmath}
\usepackage{parskip}
\definecolor{nbbg}{HTML}{f5f5f5}
\lstset{basicstyle=\ttfamily\footnotesize,backgroundcolor=\color{nbbg},
  breaklines=true,frame=single,framesep=4pt,columns=fullflexible,
  literate={±}{{$\pm$}}1 {≈}{{$\approx$}}1 {→}{{$\rightarrow$}}1 {×}{{$\times$}}1
           {λ}{{$\lambda$}}1 {ρ}{{$\rho$}}1 {θ}{{$\theta$}}1 {σ}{{$\sigma$}}1
           {²}{{\textsuperscript{2}}}1 {α}{{$\alpha$}}1 {<->}{{$\leftrightarrow$}}1}
\setcounter{secnumdepth}{0}
\title{%s}
\author{%s}
\date{%s}
\begin{document}
\maketitle
"""


def main():
    with open(SRC) as f:
        lines = f.readlines()
    title, author, date, body = parse(lines)
    title = title or "Research notebook"
    author = author or ""
    date = date or ""
    with open(OUT, "w") as f:
        f.write(PREAMBLE % (title, author, date))
        f.write("\n".join(body))
        f.write("\n\\end{document}\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
