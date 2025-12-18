# Week 9: Boolean Combination of Invariants

This tool extends the linear invariant synthesis from Week 6 to support **boolean combinations** of linear invariants.

## Features

- **Single linear invariants**: `ax + by <= c`
- **Conjunctions**: `(ax + by <= c) && (dx + ey <= f)`
- **Disjunctions**: `(ax + by <= c) || (dx + ey <= f)`
- **Mixed combinations**: Support for up to 3 combined constraints

## Installation

Requires:
- Python 3.8+
- Z3 Solver: `pip install z3-solver`
- Dafny (for validation): Follow https://dafny.org installation guide

## Usage

### Basic Usage

```bash
# Synthesize invariants for a Dafny file
python main.py benchmarks/benchmark1_conjunction.dfy

# Synthesize and insert into output file
python main.py benchmarks/benchmark1_conjunction.dfy -o output.dfy

# Run with custom parameters
python main.py benchmarks/benchmark3_three_vars.dfy -c 15 -n 4  # larger coefficients, more constraints
```

### Testing

```bash
# Run internal test suite
python main.py --test

# Run all benchmarks
python main.py --benchmark
```

### Programmatic API

```python
from boolean_invariant_synthesis import BooleanInvariantSynthesizer

synthesizer = BooleanInvariantSynthesizer(coeff_bound=10, max_constraints=3)
result = synthesizer.synthesize("benchmarks/benchmark1_conjunction.dfy")

print(f"Found {len(result.invariants)} invariants:")
for inv in result.invariants:
    print(f"  {inv}")
```

### Direct Specification API

```python
from z3_boolean_solver import Z3BooleanSolver, BoolOp

solver = Z3BooleanSolver(coeff_bound=10)

# Synthesize for: x := 0; y := 0; while x < n: x := x+1; y := y+2
inv = solver.solve_for_coefficients(
    var_names=['x', 'y'],
    init_values={'x': 0, 'y': 0},
    update_exprs={'x': '+1', 'y': '+2'},
    num_constraints=2,
    combination_type=BoolOp.AND
)

print(inv.to_string())  # e.g., "(x >= 0) && (y - 2*x <= 0)"
```

## File Structure

```
week9/
├── main.py                      # Main entry point
├── boolean_invariant_synthesis.py  # Core synthesis logic
├── z3_boolean_solver.py         # Z3 solver with boolean support
├── dafny_parser.py              # Dafny source parser
├── dafny_verifier.py            # Dafny verification interface
├── benchmarks/                  # Test programs
│   ├── benchmark1_conjunction.dfy
│   ├── benchmark2_conditional.dfy
│   ├── benchmark3_three_vars.dfy
│   ├── benchmark4_bounded.dfy
│   └── benchmark5_two_phase.dfy
└── README.md
```

## Supported Invariant Forms

### Single Constraints
```
ax + by + c <= 0
```

### Conjunctions (AND)
```
(a1*x + b1*y + c1 <= 0) && (a2*x + b2*y + c2 <= 0)
```

### Disjunctions (OR)
```
(a1*x + b1*y + c1 <= 0) || (a2*x + b2*y + c2 <= 0)
```

## Algorithm

1. **Parse** the Dafny program to extract:
   - Loop variables
   - Initial values
   - Update expressions
   - Pre/postconditions

2. **Generate constraints** for coefficient search:
   - For each reachable state (simulated iterations)
   - Constrain coefficients to satisfy invariant template

3. **Combine constraints** using boolean operators:
   - Try single constraints first
   - Then conjunctions of size 2, 3, ...
   - Then disjunctions of size 2, 3, ...

4. **Validate** with Dafny verifier (optional)

5. **Insert** valid invariants into output program

## Benchmarks

| Benchmark | Description | Expected Invariant Type |
|-----------|-------------|------------------------|
| benchmark1 | Two-variable linear | Conjunction |
| benchmark2 | Conditional update | Mixed bounds |
| benchmark3 | Three variables | Triple conjunction |
| benchmark4 | Bounded counter | Conjunction with bounds |
| benchmark5 | Two-phase loop | Multiple loops |

## Limitations

- Only handles integer linear arithmetic
- Update expressions must be linear (x := x + c)
- Maximum coefficient bound is configurable (default: 10)
- Nested loops not fully supported

## References

- Week 6: Linear Invariant Synthesis (foundation)
- CAV03: Linear Invariant Generation Using Non-Linear Constraint Solving
- CAV10: Constraint Solving for Program Verification
