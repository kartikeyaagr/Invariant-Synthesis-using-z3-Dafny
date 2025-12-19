# Dafny Invariant Synthesis Pipeline (Weeks 1-12)

Automated loop invariant synthesis for Dafny programs using constraint-based techniques.

## Overview

This project implements a complete pipeline for automatically synthesizing loop invariants for Dafny programs. It progresses through increasingly sophisticated techniques:

| Week | Component | Description |
|------|-----------|-------------|
| 1 | Setup | Environment verification (Python, Z3, Dafny) |
| 2 | Manual | Hand-written invariant examples |
| 3 | Benchmarks | Test suite of Dafny programs |
| 4 | Z3 | Constraint solving fundamentals |
| 5 | Parser | Dafny lexer and parser |
| 6 | Linear | Linear invariant synthesis (`ax + by ≤ c`) |
| 7 | Pipeline | End-to-end verification pipeline |
| 8 | Survey | Research literature review |
| 9 | Boolean | Boolean combinations (`I₁ ∧ I₂`, `I₁ ∨ I₂`) |
| 10 | Disjunctive | Path-sensitive invariants (`cond ⟹ I`) |
| 11 | Quadratic | Polynomial invariants (`ax² + bxy + cy² + ...`) |
| 12 | Arrays | Quantified invariants (`∀k :: P(a[k])`) |

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Z3 Solver
pip install z3-solver

# Dafny (optional, for verification)
# See: https://github.com/dafny-lang/dafny/wiki/INSTALL
```

### Setup

1. Place `main.py` in the same directory as your week folders:

```
your_project/
├── main.py          ← Entry point
├── week1/           ← Or Week1, w1, week01
├── week2/
├── ...
├── week12/
└── benchmarks/      ← Optional test files
```

2. Verify installation:

```bash
python main.py --list
```

## Usage

### Quick Start

```bash
# Demo all 12 weeks
python main.py --demo

# List available modules
python main.py --list

# Run all benchmarks
python main.py --test

# Synthesize invariants for a file
python main.py program.dfy

# Use ALL synthesis techniques
python main.py program.dfy --all

# JSON output
python main.py program.dfy --all --json
```

### Example Output

```
============================================================
Dafny Invariant Synthesis - Full Demo (Weeks 1-12)
============================================================

[Week 1: Environment Setup]
  Python: 3.12.4
  Z3: ✓ Installed
  Dafny: ✓ Installed
  Week folders found: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

[Week 6: Linear Invariant Synthesis]
  File: test_program.dfy
  Candidate invariant: -2 * x + 1 * y <= 0
  Successfully synthesized invariant!

[Week 9: Boolean Combinations]
  File: benchmark1_conjunction.dfy
    -> -y - x - n <= 0
    -> (-y - x - n <= 0) && (-y - x - n <= 0)

[Week 10: Disjunctive Invariants]
  File: bench1_even_odd.dfy
    -> (i % 2 == 0 ==> -n - i - x <= 0) && (!(i % 2 == 0) ==> -n - i - x <= 0)

[Week 11: Quadratic Invariants]
  File: bench1_triangular.dfy
    -> -n * n - sum * sum - n * sum - n * i - sum * i - n - sum == 0

[Week 12: Array Invariants]
  File: bench1_init.dfy
    -> 0 <= i <= a.Length
    -> forall k :: 0 <= k < i ==> a[k] == 0
```

## Week-by-Week Details

### Week 1: Environment Setup
Verifies Python, Z3, and Dafny installation.

### Week 2: Manual Invariants
Hand-written invariant examples demonstrating patterns like:
- Two-variable synchronization
- Integer division
- GCD algorithm
- Fast exponentiation

### Week 3: Benchmark Suite
Collection of Dafny programs for testing synthesis:
- `3bit_squares.dfy`
- `binary_popcount.dfy`
- `digital_sum_steps.dfy`
- And more...

### Week 4: Z3 Constraint Solving
Fundamentals of Z3 for invariant synthesis:
```python
from z3 import Solver, Int, sat
s = Solver()
x, y = Int('x'), Int('y')
s.add(x + y == 10, x > 0, y > 0)
if s.check() == sat:
    print(s.model())  # x=4, y=6
```

### Week 5: Dafny Parser
Lexer and parser for Dafny source code:
```python
from parser import DafnyExtractor
extractor = DafnyExtractor()
result = extractor.parse_file("program.dfy")
# Returns: {methods: [...], loops: [...], preconditions: [...]}
```

### Week 6: Linear Invariant Synthesis
Synthesizes invariants of form `ax + by ≤ c`:
```python
from linear_invariant_synthesis import LinearInvariantSynthesizer
synth = LinearInvariantSynthesizer()
synth.synthesize("program.dfy")
# Output: -2 * x + 1 * y <= 0
```

### Week 7: Verification Pipeline
End-to-end pipeline:
1. `program_parser.parse_dafny_program(file)` - Parse source
2. `invariant_inserter.insert_invariants(...)` - Instrument code
3. `dafny_verifier.verify_dafny_program(file)` - Run Dafny

### Week 8: Research Survey
Key papers surveyed:
- **CAV03**: Linear Invariant Generation Using Non-linear Constraint Solving (Colón et al.)
- **CAV08**: Constraint-Based Approach for Polynomial Invariants (Gulwani et al.)
- **CAV10**: Unified Framework Using ICE Learning (Sharma, Aiken)
- **CSUR14**: Loop Invariants Taxonomy (Furia et al.)

### Week 9: Boolean Combinations
Synthesizes conjunctions and disjunctions:
```
Input:  x=0, y=0; while(...) { x++; y++; }
Output: (x >= 0) && (y >= 0) && (x == y)
```

### Week 10: Disjunctive Invariants
Path-sensitive analysis for conditionals:
```
Input:  if (i % 2 == 0) x += 2 else x += 1
Output: (i % 2 == 0 ==> x == 2*i) && (i % 2 != 0 ==> x == 2*i - 1)
```

### Week 11: Quadratic Invariants
Polynomial relationships:
```
Input:  sum = 0; for i := 1 to n { sum += i; }
Output: 2 * sum == i * (i - 1)  // Triangular numbers
```

### Week 12: Array Invariants
Quantified properties over arrays:
```
Input:  while (i < a.Length) { a[i] := 0; i++; }
Output: forall k :: 0 <= k < i ==> a[k] == 0
```

## Invariant Types Supported

| Type | Template | Example |
|------|----------|---------|
| Linear | `ax + by + c ≤ 0` | `-x - y <= 0` |
| Conjunction | `I₁ ∧ I₂` | `x >= 0 && y >= 0` |
| Disjunction | `I₁ ∨ I₂` | `x < 5 \|\| y < 10` |
| Implication | `cond ⟹ I` | `i % 2 == 0 ==> x == 2*i` |
| Quadratic | `ax² + by² + cxy + ...` | `2*sum - i*i + i == 0` |
| Quantified | `∀k :: P(k)` | `forall k :: 0 <= k < i ==> a[k] == 0` |

## Algorithm Overview

The synthesis pipeline uses **template-based constraint solving**:

1. **Parse**: Extract loops, variables, and specifications from Dafny source
2. **Simulate**: Execute loop symbolically to collect reachable states
3. **Constrain**: Generate Z3 constraints requiring template to hold at all states
4. **Solve**: Find coefficients satisfying all constraints
5. **Verify**: Optionally validate with Dafny verifier

```
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌─────────┐    ┌──────────┐
│  Parse  │───▶│ Simulate │───▶│ Constrain │───▶│  Solve  │───▶│  Verify  │
└─────────┘    └──────────┘    └───────────┘    └─────────┘    └──────────┘
   Week 5         Week 6          Week 6          Week 4         Week 7
```

## Benchmarks

Run all benchmarks:
```bash
python main.py --test
```

Expected output:
```
[Week 9 Benchmarks]
  ✓ benchmark1_conjunction.dfy: 5 invariants
  ✓ benchmark2_conditional.dfy: 5 invariants
  ...

[Week 10 Benchmarks]
  ✓ bench1_even_odd.dfy: 3 invariants
  ...

[Week 11 Benchmarks]
  ✓ bench1_triangular.dfy: 4 invariants
  ...

[Week 12 Benchmarks]
  ✓ bench1_init.dfy: 5 invariants
  ...

Summary: 25/25 benchmarks passed
```

## Folder Structure

The pipeline auto-detects week folders with various naming conventions:
- `week9`, `Week9`, `w9`, `week09`

Each week folder should contain:
```
week9/
├── main.py                         # Week-specific entry point
├── boolean_invariant_synthesis.py  # Main synthesizer
├── z3_boolean_solver.py            # Z3 interface
├── dafny_parser.py                 # Parser (if needed)
└── benchmarks/                     # Test files
    ├── benchmark1_conjunction.dfy
    └── ...
```

## Troubleshooting

### "Module not available"
Ensure the week folder exists and contains the expected Python files:
```bash
python main.py --list
```

### "Z3 not installed"
```bash
pip install z3-solver
```

### "Dafny not found"
Week 6 and 7 require Dafny for validation. Install from:
https://github.com/dafny-lang/dafny/wiki/INSTALL

### Import conflicts
The pipeline uses `importlib` to load week-specific modules, avoiding conflicts between weeks with similar file names.

## References

1. Colón, M., Sankaranarayanan, S., & Sipma, H. (2003). Linear invariant generation using non-linear constraint solving. CAV.
2. Gulwani, S., Srivastava, S., & Venkatesan, R. (2008). Program analysis as constraint solving. PLDI.
3. Sharma, R., & Aiken, A. (2010). From invariant checking to invariant inference. CAV.
4. Furia, C. A., Meyer, B., & Velder, S. (2014). Loop invariants: Analysis, classification, and examples. ACM Computing Surveys.