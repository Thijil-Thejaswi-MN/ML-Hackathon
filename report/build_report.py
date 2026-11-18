"""Typeset the technical report: report/PathSnap_Technical_Report.md -> report/PathSnap_Technical_Report.pdf

    python report/build_report.py

Needs pandoc (>= 3) and a TeX distribution with XeLaTeX + latexmk (fonts: TeX Gyre Pagella, Fira Sans, DejaVu Sans Mono). The markdown file is the single source of the
report text (readable on GitHub); this script only prepares a temporary copy for typesetting:
  * removes the manual section numbers (LaTeX numbers sections, figures and pages itself),
  * gives every figure a label and turns textual figure references into live cross-references,
  * converts HTML subscripts to pandoc syntax, widens the columns of the model-comparison table,
then runs pandoc (with latex/header.tex, latex/titlepage.tex, latex/filter.lua) and latexmk -xelatex.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "PathSnap_Technical_Report.md"
OUT = HERE / "PathSnap_Technical_Report.pdf"
BUILD = HERE / "build"

# image file (substring) -> figure label
FIG_LABELS = {
    "03_missing_values": "missing", "02_sampling_intervals": "sampling", "04_label_distance": "labeldist",
    "07_trajectories": "traj", "05_network_stats": "network", "06_candidates": "cands", "map_eda_00000033": "edamap",
    "01_cv_macro_f1": "cvf1", "03_student_test_top1": "testtop1", "04_log_loss": "logloss", "06_overfitting_gap": "gap",
    "lightgbm/history": "lgbhist", "05_inference_time": "infer", "optimisation_history": "bohist",
    "param_vs_score": "boparams", "01_hmm_vs_classifier": "hmmtrip", "02_hmm_vs_classifier": "hmmoverall",
    "map_hmm_00000076": "hmmmap", "ui_two_laptops": "ui2", "ui_trip_00000076": "ui76", "api_docs": "apidocs", "api_console_laptop1": "apiconsole",
}

# textual references -> LaTeX cross-references
TEXT_REFS = [
    ("(figure 2.3)", "(Figure~\\ref{fig:labeldist})"),
    ("figure 3.3e", "Figure~\\ref{fig:lgbhist}"),
    ("(figure 5.1)", "(Figure~\\ref{fig:hmmtrip})"),
    ("(figures 2.6, 6.1, 7.2, 7.3)",
     "(Figures~\\ref{fig:edamap}, \\ref{fig:hmmmap}, \\ref{fig:ui76} and \\ref{fig:apidocs})"),
    ("figure 7.1 was taken by hand", "Figure~\\ref{fig:ui2} was taken by hand"),
]


def prepare(md: str) -> str:
    # drop YAML front matter and the opening team/repository/partition lines (they are on the cover page)
    md = re.sub(r"\A---\n.*?\n---\n", "", md, flags=re.S)
    md = md[md.index("## 1."):]
    # horizontal rules between sections are not needed in print
    md = re.sub(r"(?m)^---\s*$", "", md)
    # headings: strip manual numbers; first section gets a print title; appendix unnumbered
    md = re.sub(r"(?m)^(#{2,4}) \d+(?:\.\d+)*\.? ", r"\1 ", md)
    md = md.replace("## Problem and pipeline in one paragraph", "## Problem and approach")
    md = re.sub(r"(?m)^## Appendix .*$", "## Appendix: reproducing every result {-}", md)
    # superscript digits are not in the body font -> real superscript
    md = md.replace("10⁻⁶", "10^−6^")
    # HTML subscripts -> pandoc subscripts
    md = re.sub(r"<sub>([^<\s]+)</sub>", r"~\1~", md)
    # figures: strip manual caption numbers, add labels
    def fig(m):
        cap, path, attrs = m.group(1), m.group(2), m.group(3) or ""
        cap = re.sub(r"^(Figure \d+(\.\d+)? — |\([a-f]\) )", "", cap)
        label = next((v for k, v in FIG_LABELS.items() if k in path), None)
        if label is None:
            sys.exit(f"no label for figure {path}")
        inner = attrs.strip("{}").strip()
        if label == "ui76":
            inner = "width=68%"
        return f"![{cap}]({path}){{#fig:{label} {inner}}}"
    md = re.sub(r"!\[(.*?)\]\(([^)]+)\)(\{[^}]*\})?", fig, md)
    for old, new in TEXT_REFS:
        if old not in md:
            sys.exit(f"reference text not found: {old!r}")
        md = md.replace(old, new)
    # model-comparison table: relative column widths (Model, family, configuration wider)
    sep = "|---|---|---|---|---|---|---|---|---|"
    i = md.index(sep)
    start = md.rfind("\n\n", 0, i) + 2                      # header row of the comparison table
    end = md.index("\n\n", i)                               # end of the table
    table = md[start:end].replace(sep, "|---------|---------|-------------|-----------|------|------|------|------|------|", 1)
    md = (md[:start] + "```{=latex}\n\\begingroup\\renewcommand{\\tablefont}{\\scriptsize}\n```\n\n" + table
          + "\n\n```{=latex}\n\\endgroup\n```" + md[end:])
    # start the comparison subsection on a fresh page (avoids a longtable head being emitted before the heading)
    md = md.replace("### Overall model comparison", "```{=latex}\n\\clearpage\n```\n\n### Overall model comparison", 1)
    # per-model answers: A / B / C as separate paragraphs
    md = re.sub(r"(?m)^(\*\*[BC]\. )", r"\n\1", md)
    # screenshot sizes for print
    md = md.replace("ui_two_laptops_trip_00000007.png){#fig:ui2 width=100%}", "ui_two_laptops_trip_00000007.png){#fig:ui2 width=92%}")
    return md


def main() -> None:
    BUILD.mkdir(exist_ok=True)
    tex_md = BUILD / "report_print.md"
    tex_md.write_text(prepare(SRC.read_text(encoding="utf-8")), encoding="utf-8")
    tex = BUILD / "PathSnap_Technical_Report.tex"
    subprocess.run([
        "pandoc", str(tex_md), "-f", "markdown+pipe_tables+implicit_figures+link_attributes+subscript+raw_tex",
        "-t", "latex", "-s", "-o", str(tex), "--shift-heading-level-by=-1", "--number-sections",
        "-V", "documentclass=article", "-V", "fontsize=11pt", "-V", "colorlinks=true",
        "-H", str(HERE / "latex/header.tex"), "-B", str(HERE / "latex/titlepage.tex"),
        "--lua-filter", str(HERE / "latex/filter.lua"), "--resource-path", str(HERE),
    ], check=True, cwd=HERE)
    # image paths in the markdown are relative to report/; compile from there
    subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error",
                    f"-outdir={BUILD}", str(tex)], check=True, cwd=HERE,
                   stdout=subprocess.DEVNULL if "-q" in sys.argv else None)
    shutil.copy2(BUILD / "PathSnap_Technical_Report.pdf", OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
