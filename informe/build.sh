#!/usr/bin/env bash
set -euo pipefail

latex template.tex
dvips template.dvi
ps2pdf template.ps
