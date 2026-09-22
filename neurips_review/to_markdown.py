"""Convert reply_all.tex into the per-review markdown that gets pasted into OpenReview.

NeurIPS 2026 rebuttals are plain text with markdown, one box per review, hard
limit 10,000 characters each. No file uploads, so the .tex is a working document
only -- this script produces the thing actually submitted, and the character
count it reports is the one that counts.

Math is left as $...$: OpenReview renders MathJax. Custom macros are not
MathJax, so they are expanded first. booktabs tables are not MathJax either, so
they become markdown tables.

Usage
-----
    python neurips_review/to_markdown.py            # write + report counts
    python neurips_review/to_markdown.py --count    # report counts only

Output
------
    neurips_review/md/{ac_4ZQS,r1_1bPK,r2_Vzfg,r3_PVbi}.md
"""
import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'reply_all.tex')
OUT = os.path.join(HERE, 'md')
LIMIT = 10000

# Custom macros from the preamble -- MathJax does not know these.
MACROS = {
    r'\Rw': r'R^2_{\widehat W}',
    r'\Rt': r'R^2_{\hat\tau}',
    r'\Rv': r'R^2_{\hat V^{\mathrm{rest}}}',
}

SLUG = {
    'Area Chair 4ZQS': 'ac_4ZQS',
    'Reviewer 1bPK': 'r1_1bPK',
    'Reviewer Vzfg': 'r2_Vzfg',
    'Reviewer PVbi': 'r3_PVbi',
}


def expand_macros(t):
    for m, v in MACROS.items():
        t = re.sub(re.escape(m) + r'(?![a-zA-Z])', v.replace('\\', '\\\\'), t)
    # \Mat{W} -> \mathbf{W}
    t = re.sub(r'\\Mat\{([^}]*)\}', r'\\mathbf{\1}', t)
    return t


def convert_table(block):
    """A center/tabular block -> a markdown table.

    Keeps cells verbatim (math included); only the rules and column spec go.
    """
    m = re.search(r'\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}', block, re.S)
    if not m:
        return ''
    body = m.group(1)
    body = re.sub(r'\\(toprule|midrule|bottomrule)', '', body)
    rows = [r.strip() for r in body.split(r'\\')]
    rows = [r for r in rows if r.strip()]
    out = []
    for i, r in enumerate(rows):
        cells = [c.strip().replace('\n', ' ') for c in r.split('&')]
        cells = [re.sub(r'\s+', ' ', inline(c)).strip() for c in cells]
        out.append('| ' + ' | '.join(cells) + ' |')
        if i == 0:
            out.append('|' + '---|' * len(cells))
    # trailing {\footnotesize ...} caption after the tabular. Strip only the
    # wrapper braces -- a blanket \} removal would break $V^{\mathrm{rest}}$.
    tail = block[m.end():]
    tail = tail.replace(r'\end{center}', '')
    tail = re.sub(r'^\s*\}', '', tail)      # closes the tabular's {\footnotesize
    tail = re.sub(r'\{\\footnotesize\s*', '', tail)
    tail = re.sub(r'\}\s*$', '', tail.strip())
    tail = re.sub(r'\s+', ' ', inline(tail)).strip()
    if tail:
        out.append('')
        out.append('*' + tail + '*')
    return '\n'.join(out)


def inline(t):
    """Text-mode LaTeX -> markdown. Math stays as $...$."""
    t = re.sub(r'\\textbf\{([^{}]*)\}', r'**\1**', t)
    t = re.sub(r'\\emph\{([^{}]*)\}', r'*\1*', t)
    t = re.sub(r'\\texttt\{([^{}]*)\}', r'`\1`', t)
    t = t.replace(r'\dots', '...')
    t = re.sub(r"``(.*?)''", r'"\1"', t, flags=re.S)
    t = t.replace('---', '\u2014').replace('--', '\u2013')
    # Text-mode only: inside $...$ a bare % starts a MathJax comment and would
    # swallow the rest of the line, and \, is a valid thin space.
    def _text(p):
        p = re.sub(r'\\([%&_#])', r'\1', p)
        p = re.sub(r'\\,', ' ', p)          # thin space
        p = re.sub(r'\\(?=\s)', '', p)      # \  and \<newline> interword space
        return p
    t = ''.join(p if i % 2 else _text(p)
                for i, p in enumerate(re.split(r'(\$[^$]*\$)', t)))
    t = t.replace('~', ' ')
    t = re.sub(r'\\bigskip|\\hrule|\\vspace\{[^}]*\}|\\maketitle', '', t)
    return t


def unwrap(par):
    """Join hard-wrapped source lines into one paragraph line."""
    return re.sub(r'\s*\n\s*', ' ', par).strip()


def convert_section(body):
    # pull the center/tabular blocks out first, they are not paragraphs
    chunks = []
    pos = 0
    for m in re.finditer(r'\\begin\{center\}.*?\\end\{center\}', body, re.S):
        chunks.append(('text', body[pos:m.start()]))
        chunks.append(('table', m.group(0)))
        pos = m.end()
    chunks.append(('text', body[pos:]))

    out = []
    for kind, chunk in chunks:
        if kind == 'table':
            out.append(convert_table(chunk))
            continue
        chunk = inline(chunk)
        for par in re.split(r'\n\s*\n', chunk):
            par = par.strip()
            if not par:
                continue
            # \paragraph{X} Y -> **X** Y
            par = re.sub(r'\\paragraph\{([^{}]*)\}', r'**\1**', par)
            # \[ ... \] display math -> $$ ... $$
            par = re.sub(r'\\\[(.*?)\\\]', lambda m: '$$' + unwrap(m.group(1)) + '$$',
                         par, flags=re.S)
            out.append(unwrap(par))
    md = '\n\n'.join(x for x in out if x.strip())
    return re.sub(r'\n{3,}', '\n\n', md).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', action='store_true', help='report counts, write nothing')
    args = ap.parse_args()

    t = open(SRC).read()
    t = re.sub(r'(?m)^%.*$', '', t)          # whole-line comments
    t = re.sub(r'(?<!\\)%.*$', '', t, flags=re.M)   # trailing comments
    t = t.split(r'\begin{document}')[1].split(r'\end{document}')[0]
    t = expand_macros(t)

    parts = re.split(r'\\section\*\{([^}]*)\}', t)[1:]
    sections = list(zip(parts[0::2], parts[1::2]))

    if not args.count:
        os.makedirs(OUT, exist_ok=True)

    print(f'{"section":<20} {"chars":>7} {"spare":>7}')
    print('-' * 36)
    worst = 0
    for title, body in sections:
        md = convert_section(body)
        n = len(md)
        worst = max(worst, n)
        flag = '  OVER' if n > LIMIT else ''
        print(f'{title:<20} {n:>7,} {LIMIT - n:>7,}{flag}')
        if not args.count:
            slug = SLUG.get(title, re.sub(r'\W+', '_', title).lower())
            with open(os.path.join(OUT, slug + '.md'), 'w') as f:
                f.write(md + '\n')

    if not args.count:
        print(f'\nwrote {len(sections)} files to {OUT}')
    return 1 if worst > LIMIT else 0


if __name__ == '__main__':
    raise SystemExit(main())
