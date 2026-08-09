#!/usr/bin/env python3
"""Build a self-contained Overleaf project from the thesis sources.

Collects thesis_main.tex, every thesis_chapter_*.tex, the bibliography, and only
the figures actually referenced by an \\includegraphics in those files, into
docs/thesis_overleaf.zip. Upload that zip to Overleaf as a new project.

The zip is a build artifact and is gitignored - regenerate it, do not commit it.

Usage (from anywhere):
    python3 docs/make_overleaf_zip.py [-o OUTPUT.zip]
"""
import argparse
import os
import re
import zipfile

DOCS = os.path.dirname(os.path.abspath(__file__))
EXTS = (".pdf", ".png", ".jpg", ".jpeg")


def referenced_figures(tex_files):
    """Figure paths named by \\includegraphics across the given .tex files."""
    figs = set()
    for name in tex_files:
        src = open(os.path.join(DOCS, name), encoding="utf-8").read()
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", src):
            figs.add(m.group(1).strip())
    return figs


def resolve(ref):
    """Map an \\includegraphics argument onto a file under docs/figures/."""
    rel = ref[len("figures/"):] if ref.startswith("figures/") else ref
    candidates = [rel] if os.path.splitext(rel)[1] else [rel + e for e in EXTS]
    for c in candidates:
        path = os.path.join(DOCS, "figures", c)
        if os.path.exists(path):
            return c, path
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=os.path.join(DOCS, "thesis_overleaf.zip"))
    args = ap.parse_args()

    tex = ["thesis_main.tex"] + sorted(
        f for f in os.listdir(DOCS)
        if f.startswith("thesis_chapter_") and f.endswith(".tex"))
    bib = [f for f in ("thesis_references.bib",) if os.path.exists(os.path.join(DOCS, f))]

    resolved, missing = {}, []
    for ref in sorted(referenced_figures(tex)):
        rel, path = resolve(ref)
        if path:
            resolved[rel] = path
        else:
            missing.append(ref)

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
        for f in tex + bib:
            z.write(os.path.join(DOCS, f), f)
        for rel, path in sorted(resolved.items()):
            z.write(path, os.path.join("figures", rel))

    print(f"{args.output}")
    print(f"  {len(tex)} tex + {len(bib)} bib + {len(resolved)} figures, "
          f"{os.path.getsize(args.output) / 1e6:.1f} MB")
    if missing:
        print("  WARNING: referenced but not found under docs/figures/:")
        for m in missing:
            print(f"    {m}")
        print("  (older IROS/thesis drafts in docs/ reference figures the current"
              " thesis does not use; harmless unless thesis_main.tex needs them)")


if __name__ == "__main__":
    main()
