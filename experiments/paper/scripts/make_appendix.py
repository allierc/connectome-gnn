"""appendix.tex: the appendix of main.tex on its own, compiled to appendix.pdf.

    python scripts/make_appendix.py          (from experiments/paper; main.tex and main.aux must exist)

Copies main.tex's preamble and its text from `\\appendix` to the end (so the sections, figures, tables and
equations are those of main.pdf, with the same letters and numbers), and adds what a stand-alone compile
needs:
  - xr-hyper `\\externaldocument{main}`: \\cref / \\ref to a label of the main text (Table 1, Figure 2, the
    equations 1-4, section 3 ...) resolve through main.aux, so build main.pdf first;
  - the counters the main text leaves behind (2 figures, 2 tables, 4 equations, the appendix starting on
    page 13), so Supplementary Figure 3, Supplementary Table 3, equation 5 and the page numbers agree with
    main.pdf;
  - the bibliography, restricted by BibTeX to the references the appendix cites.
appendix.tex is generated: change main.tex (through scripts/edit_tex.py), not appendix.tex.
"""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST_APPENDIX_PAGE = 13        # main.pdf: the page of "A Supplementary Tables"
N_MAIN = {"figure": 2, "table": 2, "equation": 4}      # numbered in the main text, before \appendix

tex = open(os.path.join(HERE, "main.tex")).read()
pre, rest = tex.split("\\begin{document}", 1)
body = rest[rest.index("\\appendix"):rest.rindex("\\end{document}")].rstrip()
body = re.sub(r"(\n\s*%+\s*)?\n\s*\\newpage\s*$", "", body).rstrip()     # the closing \newpage of main.tex
body = body.replace("\\newpage\n\\section{Supplementary Tables}", "\\section{Supplementary Tables}", 1)   # the title takes that page
out = (pre.rstrip() + "\n\n% generated from main.tex by scripts/make_appendix.py -- do not edit\n"
       "\\usepackage{xr-hyper}\n\\externaldocument{main}\n\n\\begin{document}\n"
       f"\\setcounter{{page}}{{{FIRST_APPENDIX_PAGE}}}\n"
       + "".join(f"\\setcounter{{{k}}}{{{v}}}\n" for k, v in N_MAIN.items())
       + "\\begin{center}\\Large\\bfseries Graph neural networks recover interpretable circuit models from neural "
         "activity: appendix\\end{center}\n\\vspace{1em}\n\n"
       + body + "\n\n{\\small\n\\bibliographystyle{plainnat}\n\\bibliography{references,paperpile,refs}\n}\n\n\\end{document}\n")
open(os.path.join(HERE, "appendix.tex"), "w").write(out)
print(f"wrote appendix.tex ({out.count(chr(10))} lines)")

env = dict(os.environ, PATH="/workspace/.conda_envs/neural-graph-linux/bin:" + os.environ["PATH"])
run = lambda *c: subprocess.run(c, cwd=HERE, env=env, capture_output=True, text=True)
run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "appendix.tex")
run("bibtex", "appendix")
run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "appendix.tex")
res = run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "appendix.tex")
log = open(os.path.join(HERE, "appendix.log")).read()
errs = [l for l in log.splitlines() if l.startswith("!")]
undefined = sorted(set(re.findall(r"Reference `([^']*)' on page", log)))
print("errors:", errs or "none")
print("undefined references:", undefined or "none")
print(re.search(r"Output written on appendix.pdf \(\d+ pages", log).group(0) if "Output written" in log else "no pdf")
