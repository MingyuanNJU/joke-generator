#!/usr/bin/env python3
"""
Automatic Paper Reproduction Machine
=====================================
Analyzes a quantum physics / condensed matter PDF paper using Claude,
produces a structured report, and generates runnable Python simulation code.

Usage:
    python paper_reproducer.py <paper.pdf> [--output-dir <dir>]

Requires:
    ANTHROPIC_API_KEY environment variable
    pip install anthropic
"""

import argparse
import base64
import os
import sys
from pathlib import Path

import anthropic

# ── Constants ────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-6"
MAX_PDF_BYTES = 32 * 1024 * 1024  # 32 MB hard limit for the API

SYSTEM_PROMPT = """You are an expert theoretical and computational physicist specializing in \
quantum optics and condensed matter physics. You have deep expertise in:

- Open quantum systems: Lindblad master equations, quantum trajectories, Floquet theory
- Many-body physics: Hubbard models, Heisenberg spin chains, BCS superconductivity, Kondo physics
- Numerical methods:
  * Master equation solvers (qutip mesolve, mcsolve)
  * 4th-order Runge-Kutta (RK4) time integration
  * Exact diagonalization (ED) via scipy.sparse or QuSpin
  * Density Matrix Renormalization Group (DMRG) via ITensor or QuSpin
  * Quantum Monte Carlo (QMC): SSE, DQMC
- Software packages: QuTiP, QuSpin, ITensor (Julia), NumPy, SciPy, Matplotlib

Your analyses are precise, identify the core physics and numerics, and your simulation \
code is clean, well-commented, and immediately runnable."""

ANALYSIS_PROMPT = """Please analyze the attached PDF paper thoroughly and produce a structured \
report in Markdown. The report must contain all of the following sections:

# Paper Analysis Report

## 1. Summary
One-paragraph overview of the paper's contributions and main claims.

## 2. Physical Model & Theory
- The system Hamiltonian (write it out explicitly with all terms defined)
- Key approximations and their physical justification
- Relevant energy/time/length scales
- Symmetries exploited

## 3. Key Equations & Derivations
List and explain the central equations (equation of motion, self-energy, \
effective Hamiltonian, partition function, etc.). Note which are derived \
analytically vs. taken from prior work.

## 4. Numerical Methods & Algorithms
- Exact method(s) used (e.g., master equation + RK4, ED, DMRG, QMC)
- Basis size / system size / Hilbert-space dimension
- Convergence criteria and parameters (bond dimension, Trotter step, MC sweeps, etc.)
- Any symmetry sectors or truncation schemes

## 5. Main Numerical Results
Describe each key figure/table:
- What quantity is plotted vs. what
- Physical regime (parameters, temperature, coupling strength)
- Qualitative behavior and the physical interpretation
- Any phase boundaries, critical points, or scaling behavior

## 6. Recommended Implementation Approach
Based on the above, specify:
- Best Python package for reproduction (qutip / quspin / scipy / custom)
- Primary algorithm to implement
- Key parameters to reproduce the central result
- Expected computational cost (feasible on a laptop? cluster needed?)

Be precise and technical. Do not omit equations or numerical details."""

CODE_PROMPT_TEMPLATE = """Based on the paper analysis report below, write a complete, \
runnable Python simulation script that reproduces the paper's central numerical result.

<report>
{report}
</report>

Requirements for the simulation code:
1. **Package selection**: Choose from qutip, quspin, numpy/scipy based on what the \
   analysis recommends. Import only what is needed.
2. **Algorithm**: Implement the exact numerical method described in the paper \
   (master equation solver, ED, DMRG-like via quspin, RK4, etc.).
3. **Structure**: The script must have:
   - A `# Parameters` section with all tunable constants (clearly commented with units)
   - A Hamiltonian / Liouvillian construction section
   - The main computation (time evolution / diagonalization / etc.)
   - A results section that prints key numerical values
   - A matplotlib plot of the main result (saved to `simulation_output.png`)
4. **Comments**: Add docstrings and inline comments explaining the physics at each step.
5. **Self-contained**: The script must run with only `pip install <package>` — no extra \
   data files required. If parameters are approximate (paper doesn't give exact values), \
   state that in a comment and use physically reasonable defaults.
6. **Error handling**: Wrap the main computation in a try/except and print a helpful \
   message if a required package is missing.

Output ONLY the Python script — no markdown fences, no explanatory text before or after.
Start the file with a shebang line and a module docstring explaining what it reproduces."""


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_pdf(pdf_path: Path) -> str:
    """Read a PDF file and return its base64-encoded content."""
    if not pdf_path.exists():
        sys.exit(f"Error: file not found: {pdf_path}")
    if not pdf_path.suffix.lower() == ".pdf":
        sys.exit(f"Error: expected a .pdf file, got: {pdf_path}")

    data = pdf_path.read_bytes()
    size_mb = len(data) / 1024 / 1024
    if len(data) > MAX_PDF_BYTES:
        sys.exit(
            f"Error: PDF is {size_mb:.1f} MB, exceeding the {MAX_PDF_BYTES // (1024*1024)} MB limit."
        )
    if size_mb > 10:
        print(f"Warning: PDF is {size_mb:.1f} MB — analysis may be slow.", file=sys.stderr)

    return base64.standard_b64encode(data).decode("utf-8")


def stream_and_collect(client: anthropic.Anthropic, label: str, **create_kwargs) -> str:
    """Stream a Claude response to stdout and return the full text."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}\n")

    full_text = ""
    with client.messages.stream(**create_kwargs) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_text += text

    print()  # newline after stream ends
    return full_text


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a quantum physics paper and generate simulation code."
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
    stem = args.pdf.stem  # e.g. "my_paper" from "my_paper.pdf"
    report_path = args.output_dir / f"{stem}_report.md"
    code_path = args.output_dir / f"{stem}_simulation.py"

    client = anthropic.Anthropic(api_key=api_key)

    # ── Phase 1: Paper analysis ───────────────────────────────────────────────
    print(f"\nLoading PDF: {args.pdf} …")
    pdf_b64 = load_pdf(args.pdf)

    report = stream_and_collect(
        client,
        label="Phase 1 / 2 — Analyzing paper …",
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

    report_path.write_text(report, encoding="utf-8")
    print(f"\n[Report saved → {report_path}]")

    # ── Phase 2: Code generation ──────────────────────────────────────────────
    code = stream_and_collect(
        client,
        label="Phase 2 / 2 — Generating simulation code …",
        model=MODEL,
        max_tokens=8192,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": CODE_PROMPT_TEMPLATE.format(report=report),
            }
        ],
    )

    # Strip accidental markdown fences if Claude added them anyway
    if code.startswith("```"):
        lines = code.splitlines()
        # drop first line (``` or ```python) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        code = "\n".join(lines)

    code_path.write_text(code, encoding="utf-8")
    print(f"[Simulation code saved → {code_path}]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"""
{'='*70}
  Done!
{'='*70}
  Report    : {report_path}
  Simulation: {code_path}

Next steps:
  1. Read the report to understand the paper's theory and methods.
  2. Install any required packages listed at the top of the simulation file.
  3. Run:  python {code_path}
{'='*70}
""")


if __name__ == "__main__":
    main()
