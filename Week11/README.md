# Week 11: Non-linear Invariants - Quadratic Forms

This tool synthesizes **quadratic invariants** of the form:

```
ax² + by² + cxy + dx + ey + f ≤ 0
```

Extends the linear synthesis from Weeks 6, 9-10 to handle polynomial relationships.

## Features

- **Quadratic Inequalities**: `ax² + by² + cxy + dx + ey + f ≤ 0`
- **Quadratic Equalities**: `ax² + by² + cxy + dx + ey + f == 0`
- **Pure Quadratic**: Terms without linear components
- **Cross-term Support**: `xy` product terms
- **Specialized Patterns**: Triangular numbers, square computation

## Installation

```bash
pip install z3-solver
```

## Usage

### Basic Usage

```bash
# Synthesize quadratic invariants
python main.py benchmarks/bench1_triangular.dfy

# With custom bounds
python main.py benchmarks/bench5_quadratic_growth.dfy -c 10 -l 20

# Output to file
python main.py program.dfy -o output.dfy
```

### Testing

```bash
# Run tests
python main.py --test

# Run benchmarks
python main.py --benchmark
```

### Options

```
-c, --coeff-bound N   Coefficient bound (default: 5)
-l, --loop-bound N    Loop simulation iterations (default: 15)
-o, --output FILE     Output file with inserted invariants
--json                Output as JSON
--test                Run test suite
--benchmark           Run benchmarks
```

## Quadratic Forms Supported

### 1. Pure Quadratic
```
ax² + by² + f ≤ 0
```

### 2. Mixed Quadratic
```
ax² + by² + dx + ey + f ≤ 0
```

### 3. Cross-term Quadratic
```
ax² + by² + cxy + dx + ey + f ≤ 0
```

### 4. Quadratic Equality
```
ax² + by² + cxy + dx + ey + f == 0
```

## Benchmarks

| Benchmark | Pattern | Quadratic Invariant |
|-----------|---------|---------------------|
| bench1_triangular | Triangular sum | `2*sum == i*i - i` |
| bench2_square | Square via odds | `sq == i*i` |
| bench3_product | Product | `prod == i*b` |
| bench4_sum_squares | Sum of squares | `sum >= 0` (bound) |
| bench5_quadratic_growth | y = x² | `y == x*x` |
| bench6_distance | Squared distance | `dist == x*x + y*y` |

## Algorithm

### Phase 1: Template Generation

Create coefficient variables for the quadratic template:
```
a_x² * x² + a_y² * y² + a_xy * x*y + b_x * x + b_y * y + c
```

### Phase 2: State Simulation

Simulate loop execution to collect reachable states:
```python
states = [(x₀, y₀), (x₁, y₁), ..., (xₙ, yₙ)]
```

### Phase 3: Constraint Generation

For each reachable state, add constraint:
```
a_x² * xᵢ² + a_y² * yᵢ² + ... + c ≤ 0
```

### Phase 4: Z3 Solving

Solve for coefficient values that satisfy all constraints.

## File Structure

```
week11/
├── main.py                    # Entry point
├── quadratic_synthesis.py     # Main synthesis module
├── z3_quadratic_solver.py     # Z3 solver for quadratics
├── benchmarks/                # Test programs
│   ├── bench1_triangular.dfy  # Triangular numbers
│   ├── bench2_square.dfy      # Square computation
│   ├── bench3_product.dfy     # Product via addition
│   ├── bench4_sum_squares.dfy # Sum of squares
│   ├── bench5_quadratic_growth.dfy # y = x²
│   └── bench6_distance.dfy    # Squared distance
├── tests/
│   └── test_quadratic.py
└── README.md
```

## API Example

```python
from quadratic_synthesis import QuadraticInvariantSynthesizer

synthesizer = QuadraticInvariantSynthesizer(coeff_bound=5)

# From specification
invariants = synthesizer.synthesize_from_spec(
    var_names=["x", "y"],
    init_values={"x": 0, "y": 0},
    updates={"x": "+1", "y": "+2"}
)

# Specialized: triangular numbers
invariants = synthesizer.synthesize_for_triangular("i", "sum")
# Returns: ["i * i - i - 2 * sum == 0"]
```

## Comparison with Previous Weeks

| Feature | Week 6 | Week 9-10 | Week 11 |
|---------|--------|-----------|---------|
| Linear | ✓ | ✓ | ✓ |
| Boolean combinations | - | ✓ | ✓ |
| Disjunctive | - | ✓ | ✓ |
| Quadratic | - | - | ✓ |
| Cross-terms (xy) | - | - | ✓ |
| Equality invariants | - | - | ✓ |

## Limitations

- Coefficient bound limits discovery of large-coefficient invariants
- Higher-degree polynomials (cubic+) not supported
- Loop simulation may miss complex behaviors
- Non-linear updates (x := x * x) have limited support

## Theoretical Background

Based on:
- **CAV03**: Linear invariant generation via constraint solving
- **CAV08**: Polynomial invariant synthesis using Positivstellensatz
- **Farkas' Lemma**: Extended to polynomial cone constraints

The quadratic case uses direct enumeration over reachable states rather than symbolic reasoning, which is more practical for bounded loops.

## References

- Week 6: Linear Invariant Synthesis
- Week 9: Boolean Combinations
- Week 10: Disjunctive Invariants
- CAV08: Constraint-Based Approach to Hybrid Systems
