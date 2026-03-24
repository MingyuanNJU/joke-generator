#!/usr/bin/env python3
"""
Automatic Paper Reproduction Machine
======================================
Analyzes a quantum physics / condensed matter PDF paper using Claude and
produces a comprehensive educational Jupyter notebook that integrates:
  - LaTeX derivations and equations
  - Step-by-step physical reasoning
  - Runnable simulation code (qutip / quspin / scipy)
  - Exercises for deeper understanding

Usage:
    python paper_reproducer.py <paper.pdf> [--output-dir <dir>]

Requires:
    ANTHROPIC_API_KEY environment variable
    pip install anthropic nbformat
"""

import argparse
import base64
import os
import re
import sys
from pathlib import Path

import anthropic
import nbformat
import nbformat.v4 as nbv4

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-6"
MAX_PDF_BYTES = 32 * 1024 * 1024  # 32 MB

SYSTEM_PROMPT = """You are an expert theoretical and computational physicist and an \
outstanding science educator. You specialize in quantum optics and condensed matter \
physics, and you are passionate about making complex ideas clear and accessible.

Your expertise covers:
- Open quantum systems: Lindblad master equations, quantum trajectories, Floquet theory
- Many-body physics: Hubbard models, Heisenberg spin chains, BCS superconductivity
- Numerical methods:
  * Master equation solvers (qutip mesolve / mcsolve)
  * 4th-order Runge-Kutta (RK4) integration
  * Exact diagonalization (ED) via scipy.sparse or QuSpin
  * DMRG via QuSpin
  * Quantum Monte Carlo (SSE, DQMC)
- Packages: QuTiP, QuSpin, NumPy, SciPy, Matplotlib

When teaching, you:
- Show every non-trivial derivation step by step
- Explain the physical intuition behind every equation
- Connect abstract formalism to concrete numerical implementation
- Anticipate common confusions and address them proactively"""

# ── Phase 1: deep analysis prompt ─────────────────────────────────────────────

ANALYSIS_PROMPT = """Please read the attached paper carefully and produce a thorough \
technical analysis in Markdown. This analysis will later be used to build an educational \
Jupyter notebook, so be as detailed as possible.

# Technical Analysis

## 1. Core Contribution
What does this paper claim to show? What is new compared to prior work?

## 2. Physical Model
Write out the full Hamiltonian (or Liouvillian) with all terms defined. \
List all parameters with their physical meaning and typical values used in the paper. \
State all approximations made and justify them physically.

## 3. Theoretical Framework
Reproduce the key derivations step by step (use LaTeX math). Include:
- How the effective model is derived from the microscopic one (if applicable)
- Key identities, commutation relations, or symmetry arguments used
- Any mean-field, perturbative, or RG steps

## 4. Numerical Methods
- Exact method(s) (e.g., master equation + RK4, ED, DMRG, QMC, TEBD)
- Basis choice and Hilbert space structure
- System sizes / bond dimensions / convergence parameters used
- Any symmetry sectors exploited to reduce cost

## 5. Key Results (one subsection per main figure/table)
For each:
- What quantity is plotted vs. what axis
- The parameter regime (coupling strength, temperature, system size)
- What the curve/surface shows qualitatively and what that means physically
- The most important takeaway the authors draw from this result

## 6. Implementation Roadmap
Recommend:
- Best Python package for reproduction and why
- Which result to target (good balance of insight vs. compute cost)
- Exact parameter values needed
- Estimated runtime on a modern laptop
- Any numerical pitfalls to avoid"""

# ── Phase 2: notebook generation prompt ───────────────────────────────────────

NOTEBOOK_PROMPT_TEMPLATE = """Using the technical analysis below, create a comprehensive \
educational Jupyter notebook that guides a graduate student through understanding \
AND reproducing the paper's central result from scratch.

<analysis>
{analysis}
</analysis>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — strict rules:

Every cell in the notebook must be wrapped in one of two delimiters:

  ===MARKDOWN===
  (markdown text, may contain LaTeX with $...$ or $$...$$)
  ===END===

  ===CODE===
  (valid Python code)
  ===END===

Do NOT output any text outside these delimiters.
Do NOT use triple-backtick fences inside CODE cells.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIRED NOTEBOOK STRUCTURE (produce cells in this order):

1. MARKDOWN — Title, authors (if known), one-paragraph abstract in your own words.

2. CODE — All imports. Add a comment explaining what each package is used for. \
   Include a try/except that prints a friendly message if a package is missing.

3. MARKDOWN — **Physical Setup**: describe the system in words and with a schematic \
   (ASCII art is fine). Motivate why this model is interesting.

4. MARKDOWN — **Hamiltonian / Liouvillian**: write every term in LaTeX. \
   Define every symbol. Explain the physics of each term.

5. MARKDOWN — **Key Derivation**: reproduce the most important theoretical result \
   step by step in LaTeX. Explain every line. State what assumptions are used.

6. MARKDOWN — **Numerical Method**: explain the algorithm (master equation, ED, DMRG, \
   RK4, …) in plain language. Describe the computational complexity.

7. CODE — Parameter definitions. One variable per line with an inline comment giving \
   units and the value from the paper.

8. [For each logical block of the implementation, produce a pair:]
   MARKDOWN — explain what the next code cell will do and why
   CODE      — the implementation

   Typical blocks (adapt to the paper):
   a. Construct basis / operators
   b. Build the Hamiltonian (or Liouvillian)
   c. Time-evolve or diagonalize
   d. Compute observables
   e. Plot results (save to "simulation_output.png")

9. MARKDOWN — **Results & Interpretation**: what the plot shows, whether it matches \
   the paper's figure, physical interpretation.

10. MARKDOWN — **Further Exploration**: 3–5 concrete exercises (change a parameter, \
    compute a different observable, extend to a larger system, etc.).

Make the explanations rich and pedagogical. A student reading this notebook should \
come away understanding both the physics AND the numerics deeply."""

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pdf(pdf_path: Path) -> str:
    """Read a PDF and return its base64-encoded content."""
    if not pdf_path.exists():
        sys.exit(f"Error: file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        sys.exit(f"Error: expected a .pdf file, got: {pdf_path}")
    data = pdf_path.read_bytes()
    size_mb = len(data) / 1024 / 1024
    if len(data) > MAX_PDF_BYTES:
        sys.exit(f"Error: PDF is {size_mb:.1f} MB, exceeding the {MAX_PDF_BYTES // (1024*1024)} MB limit.")
    if size_mb > 10:
        print(f"Warning: PDF is {size_mb:.1f} MB — analysis may be slow.", file=sys.stderr)
    return base64.standard_b64encode(data).decode("utf-8")


def stream_and_collect(client: anthropic.Anthropic, label: str, **create_kwargs) -> str:
    """Stream a Claude response to stdout and return the full accumulated text."""
    print(f"\n{'═'*68}")
    print(f"  {label}")
    print(f"{'═'*68}\n")
    full_text = ""
    with client.messages.stream(**create_kwargs) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_text += text
    print()
    return full_text


def parse_cells(raw: str) -> list[dict]:
    """
    Parse the delimited output into a list of cell dicts:
      {"type": "markdown" | "code", "source": str}

    Handles minor formatting variations Claude may produce.
    """
    pattern = re.compile(
        r"===\s*(MARKDOWN|CODE)\s*===\s*(.*?)\s*===\s*END\s*===",
        re.DOTALL | re.IGNORECASE,
    )
    cells = []
    for match in pattern.finditer(raw):
        cell_type = match.group(1).strip().lower()
        source = match.group(2).strip()
        if source:
            cells.append({"type": cell_type, "source": source})

    if not cells:
        # Fallback: treat the entire output as a single markdown cell
        print("\nWarning: could not parse cell delimiters; wrapping output in a single markdown cell.",
              file=sys.stderr)
        cells = [{"type": "markdown", "source": raw.strip()}]

    return cells


def build_notebook(cells: list[dict]) -> nbformat.NotebookNode:
    """Assemble an nbformat 4 notebook from parsed cells."""
    nb = nbv4.new_notebook()
    nb.metadata.update({
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    })
    nb_cells = []
    for cell in cells:
        if cell["type"] == "markdown":
            nb_cells.append(nbv4.new_markdown_cell(cell["source"]))
        else:
            nb_cells.append(nbv4.new_code_cell(cell["source"]))
    nb.cells = nb_cells
    return nb


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a quantum physics paper and generate an educational "
            "Jupyter notebook with derivations, theory, and runnable code."
        )
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF paper")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output files (default: current directory)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.pdf.stem
    analysis_path = args.output_dir / f"{stem}_analysis.md"
    notebook_path = args.output_dir / f"{stem}_notebook.ipynb"

    client = anthropic.Anthropic(api_key=api_key)

    # ── Phase 1: deep paper analysis ─────────────────────────────────────────
    print(f"\nLoading PDF: {args.pdf} …")
    pdf_b64 = load_pdf(args.pdf)

    analysis = stream_and_collect(
        client,
        label="Phase 1 / 2 — Deep paper analysis …",
        model=MODEL,
        max_tokens=8192,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {"type": "text", "text": ANALYSIS_PROMPT},
                ],
            }
        ],
    )

    analysis_path.write_text(analysis, encoding="utf-8")
    print(f"\n[Analysis saved → {analysis_path}]")

    # ── Phase 2: educational notebook generation ──────────────────────────────
    raw_notebook = stream_and_collect(
        client,
        label="Phase 2 / 2 — Building educational notebook …",
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": NOTEBOOK_PROMPT_TEMPLATE.format(analysis=analysis),
            }
        ],
    )

    cells = parse_cells(raw_notebook)
    nb = build_notebook(cells)

    with notebook_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_md = sum(1 for c in cells if c["type"] == "markdown")
    n_code = sum(1 for c in cells if c["type"] == "code")
    print(f"""
{'═'*68}
  Done!
{'═'*68}
  Analysis (reference) : {analysis_path}
  Notebook             : {notebook_path}
  Cells generated      : {n_md} markdown  +  {n_code} code

Next steps:
  pip install jupyter qutip quspin numpy scipy matplotlib
  jupyter notebook {notebook_path}
{'═'*68}
""")


if __name__ == "__main__":
    main()
