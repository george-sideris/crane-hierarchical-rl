#!/usr/bin/env python3
"""Generate expanded results table PNG for IROS 2026 supplementary video.

Produces a standalone LaTeX table, compiles to PDF, converts to 300 DPI PNG.
Output: crane_testbed/media/results_table.png (overwrite)
"""

import os
import subprocess
import tempfile
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_PNG = os.path.join(REPO_ROOT, "crane_testbed", "media", "results_table.png")

LATEX_DOC = r"""
\documentclass[border=12pt]{standalone}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{array}

\begin{document}

\renewcommand{\arraystretch}{1.25}
\setlength{\tabcolsep}{6pt}

\begin{tabular}{llccccccc}
\toprule
Method & Obs. & $g$ & $\alpha$ & $\varsigma$ & Grasp succ.\,(\%) & Clear rate\,(\%) & Knocked off\,(\%) & $c_{95}$ \\
\midrule
\textcolor{gray}{\textit{Heuristic}} & \textcolor{gray}{\textit{Pose}} & \textcolor{gray}{\textit{9.94\,$\pm$\,1.17}} & \textcolor{gray}{\textit{0.918\,$\pm$\,0.031}} & \textcolor{gray}{\textit{0.797\,$\pm$\,0.018}} & \textcolor{gray}{\textit{88.7\,$\pm$\,7.8}} & \textcolor{gray}{\textit{99.0\,$\pm$\,1.2}} & \textcolor{gray}{\textit{1.0\,$\pm$\,1.1}} & \textcolor{gray}{\textit{18.8\,$\pm$\,2.8}} \\
\addlinespace
RL & Pose (128D) & 8.51\,$\pm$\,1.27 & 0.874\,$\pm$\,0.054 & 0.958\,$\pm$\,0.014 & 58.7\,$\pm$\,10.4 & 73.3\,$\pm$\,7.2 & 0.6\,$\pm$\,0.6 & N/A \\
RL & Seg.\ PC    & 7.65\,$\pm$\,0.98 & 0.772\,$\pm$\,0.065 & 0.737\,$\pm$\,0.044 & 47.4\,$\pm$\,6.8 & 53.4\,$\pm$\,3.1 & 0.5\,$\pm$\,0.5 & N/A \\
RL & Raw PC      & 6.78\,$\pm$\,1.01 & 0.758\,$\pm$\,0.072 & 0.815\,$\pm$\,0.039 & 53.3\,$\pm$\,8.7 & 53.0\,$\pm$\,3.9 & 0.6\,$\pm$\,0.6 & N/A \\
\addlinespace
BC & Seg.\ PC    & 9.64\,$\pm$\,1.05 & 0.907\,$\pm$\,0.028 & 0.816\,$\pm$\,0.020 & 70.3\,$\pm$\,10.6 & 96.5\,$\pm$\,4.7 & 0.9\,$\pm$\,1.0 & 21.7\,$\pm$\,4.6 \\
BC & Raw PC      & \textbf{9.32\,$\pm$\,1.12} & 0.904\,$\pm$\,0.037 & 0.824\,$\pm$\,0.025 & 84.5\,$\pm$\,9.8 & \textbf{98.5\,$\pm$\,5.1} & 0.7\,$\pm$\,0.7 & \textbf{20.6\,$\pm$\,3.0} \\
\addlinespace
BC$\to$RL & Raw PC & 8.65\,$\pm$\,0.97 & \textbf{0.906\,$\pm$\,0.027} & \textbf{0.947\,$\pm$\,0.016} & \textbf{86.3\,$\pm$\,8.5} & 98.2\,$\pm$\,1.5 & 1.2\,$\pm$\,0.8 & 21.9\,$\pm$\,3.2 \\
\bottomrule
\addlinespace[4pt]
\multicolumn{9}{l}{\footnotesize $g$: mean logs grasped per episode \quad $\alpha$: grapple-to-log angular alignment (grasp-count-weighted mean)} \\
\multicolumn{9}{l}{\footnotesize $\varsigma$: grapple levelness after lift (mean over successful grasps) \quad $c_{95}$: grasp cycles to clear 95\% of pile} \\
\multicolumn{9}{l}{\footnotesize Grasp succ.: fraction of grasps with $g > 0$ \quad Clear rate: fraction of logs intentionally removed \quad Knocked off: fraction knocked outside bounds} \\
\multicolumn{9}{l}{\footnotesize All values: mean $\pm$ std over 100 episodes with randomized 200-log piles. \textbf{Bold} = best among learned methods with raw depth point cloud input per column.} \\
\multicolumn{9}{l}{\footnotesize \textit{Heuristic} row: oracle reference (uses privileged pose state).} \\
\end{tabular}

\end{document}
"""


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "table.tex")
        with open(tex_path, "w") as f:
            f.write(LATEX_DOC)

        # Compile LaTeX to PDF
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "table.tex"],
            cwd=tmpdir, capture_output=True, text=True,
        )
        pdf_path = os.path.join(tmpdir, "table.pdf")
        if not os.path.exists(pdf_path):
            print("LaTeX compilation failed:")
            print(result.stdout)
            print(result.stderr)
            return

        # Convert PDF to PNG at 300 DPI
        subprocess.run(
            ["pdftoppm", "-png", "-r", "300", "-singlefile", pdf_path,
             os.path.join(tmpdir, "table")],
            check=True,
        )
        png_path = os.path.join(tmpdir, "table.png")
        if not os.path.exists(png_path):
            print("PDF to PNG conversion failed")
            return

        shutil.copy2(png_path, OUT_PNG)
        print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
