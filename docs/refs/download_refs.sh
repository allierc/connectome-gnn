#!/usr/bin/env bash
# Download the PDFs of the papers cited in docs/zebrafish.tex into this dir.
#
# Status from the sandboxed run (2026-06-10):
#   OK (open access, fetched automatically):
#     khona2022.pdf      arXiv 2112.03978
#     goncalves2014.pdf  Frontiers (CC BY)
#     kim2017.pdf        institutional copy (UCSD course page)
#   NOT fetched here — publisher bot-wall / paywall in the sandbox.
#   Run this on a machine with normal browser access or an institutional
#   proxy (or just open the DOI and "Save as PDF"). DOIs below.
#
# cx2026 (Allier et al., companion ms in preparation) has no public PDF.

set -u
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
dl () { echo ">> $2"; curl -fL --retry 2 --max-time 120 -A "$UA" -e "$3" -o "$2" "$1" \
        && head -c5 "$2" | grep -q '%PDF' && echo "   OK $(du -h "$2"|cut -f1)" \
        || { echo "   FAIL (open the DOI manually)"; rm -f "$2"; }; }

# --- open access (should work anywhere) ---
dl "https://arxiv.org/pdf/2112.03978"                                  khona2022.pdf     "https://arxiv.org/"
dl "https://www.frontiersin.org/articles/10.3389/fncir.2014.00010/pdf" goncalves2014.pdf "https://www.frontiersin.org/"
dl "https://neurophysics.ucsd.edu/courses/physics_171/kim_jayaraman_science_2017.pdf" kim2017.pdf "https://neurophysics.ucsd.edu/"

# --- paywalled / bot-walled: DOIs (need subscription or institutional access) ---
#   petrucco2023   https://doi.org/10.1038/s41593-023-01308-5   Nat. Neurosci. (PMC10166860)
#   dana2019       https://doi.org/10.1038/s41592-019-0435-6    Nat. Methods
#   feierstein2023 https://doi.org/10.1016/j.cub.2023.07.075    Curr. Biol.
#   dunn2016artr   https://doi.org/10.7554/eLife.12741          eLife (CC BY; elifesciences.org PDF)
#   seung1996      https://doi.org/10.1073/pnas.93.23.13339     PNAS (PMC24094, public)
#   major2004      https://doi.org/10.1073/pnas.0401970101      PNAS (PMC419676, public)
#   chaudhuri2019  https://doi.org/10.1038/s41593-019-0460-x    Nat. Neurosci.
#   lyu2022        https://doi.org/10.1038/s41586-021-04067-0   Nature (PMC11104186)
#   yang2024       https://doi.org/10.1038/s41586-024-07867-2   Nature (PMC11464381)
echo
echo "Paywalled/bot-walled — fetch via DOI on an authenticated machine (see comments above)."
