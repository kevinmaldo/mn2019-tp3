#!/usr/bin/env bash
set -euo pipefail

pdflatex template.tex
#dvips template.dvi
#ps2pdf template.ps
pdflatex template.tex
#dvips template.dvi
#ps2pdf template.ps
