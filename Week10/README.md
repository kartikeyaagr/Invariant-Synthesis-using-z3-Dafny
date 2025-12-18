# Week 10: Disjunctive Invariant Synthesis

This tool synthesizes **disjunctive invariants** for programs with multiple execution paths, particularly loops containing conditional statements.

## Key Features

- **Path Analysis**: Automatically identifies execution paths through conditional branches
- **Disjunctive Invariants**: `inv1 || inv2 || ...`
- **Path-Sensitive Invariants**: `(cond1 ==> inv1) && (cond2 ==> inv2)`
- **Alternating Behavior Support**: Handles even/odd iteration patterns

## Installation

```bash
pip install z3-solver
```

## Usage

### Basic Usage

```bash
# Synthesize disjunctive invariants
python main.py benchmarks/bench1_even_odd.dfy

# Output with invariants inserted
python main.py benchmarks/bench1_even_odd.dfy -o output.dfy

# Only analyze execution paths (no synthesis)
python main.py benchmarks/bench1_even_odd.dfy --analyze
```

### Testing

```bash
# Run internal tests
python main.py --test

# Run all benchmarks
python main.py --benchmark
```

### Options

```
-o, --output FILE      Write output with invariants
-c, --coeff-bound N    Coefficient bound (default: 10)
-d, --max-disjuncts N  Maximum disjuncts (default: 3)
--analyze              Only analyze paths, no synthesis
--json                 Output as JSON
--test                 Run test suite
--benchmark            Run benchmarks
```

## Supported Patterns

### 1. If-Then-Else Branches

```dafny
while i < n {
  if (cond) {
    x := x + 2;
  } else {
    x := x + 1;
  }
  i := i + 1;
}
```

**Synthesizes**: `(cond ==> inv1) && (!cond ==> inv2)`

### 2. Threshold-Based Behavior

```dafny
while i < n {
  if (i < mid) {
    x := x + 1;
  } else {
    y := y + 1;
  }
  i := i + 1;
}
```

**Synthesizes**: `(i < mid ==> x_inv) && (i >= mid ==> y_inv)`

### 3. Conditional Accumulation

```dafny
while i < n {
  if (i % 2 == 0) {
    sum := sum + i;
  }
  i := i + 1;
}
```

**Synthesizes**: Path-sensitive invariant for accumulator

### 4. Flag-Based State Changes

```dafny
while i < n {
  if (!flag) {
    x := x + 1;
    if (x >= threshold) { flag := true; }
  } else {
    y := y + 1;
  }
  i := i + 1;
}
```

**Synthesizes**: `(!flag && x_inv) || (flag && y_inv)`

## File Structure

```
week10/
├── main.py                    # Entry point
├── disjunctive_synthesis.py   # Main synthesis logic
├── z3_disjunctive_solver.py   # Z3 solver for disjunctions
├── path_analyzer.py           # Execution path analysis
├── benchmarks/                # Test programs
│   ├── bench1_even_odd.dfy
│   ├── bench2_threshold.dfy
│   ├── bench3_sign_check.dfy
│   ├── bench4_three_way.dfy
│   ├── bench5_conditional_acc.dfy
│   └── bench6_flag_state.dfy
├── tests/
│   └── test_disjunctive.py
└── README.md
```

## Algorithm Overview

### Phase 1: Path Analysis

1. Parse loop body for conditional statements
2. Identify each execution path (if-branch, else-branch)
3. Extract variable updates per path
4. Classify loop type (linear, if-then, if-then-else, multi-branch)

### Phase 2: Synthesis Strategy Selection

- **Linear** (no conditionals): Simple linear invariant
- **If-Then-Else**: Disjunctive or path-sensitive
- **Multi-Branch**: Multiple disjuncts

### Phase 3: Constraint Generation

For each path P with guard G and updates U:
1. Simulate execution along P
2. Generate constraints: `G ==> linear_constraint`
3. Combine: `(G1 ==> inv1) && (G2 ==> inv2) && ...`

### Phase 4: Z3 Solving

- Search for coefficients satisfying all path constraints
- Return disjunctive or path-sensitive invariant

## Benchmarks

| Benchmark | Pattern | Paths | Description |
|-----------|---------|-------|-------------|
| bench1 | Even/Odd | 2 | Different updates for even/odd iterations |
| bench2 | Threshold | 2 | Behavior changes at midpoint |
| bench3 | Sign check | 2 | Update based on value sign |
| bench4 | Three-way | 3 | Modulo-3 based branching |
| bench5 | Conditional acc | 2 | Accumulate only on condition |
| bench6 | Flag state | 2 | Flag-controlled phase change |

## Comparison with Week 9

| Feature | Week 9 | Week 10 |
|---------|--------|---------|
| Boolean combinations | ✓ | ✓ |
| Path analysis | ✗ | ✓ |
| Conditional handling | Limited | Full |
| Disjunctive synthesis | Basic | Advanced |
| Path-sensitive invariants | ✗ | ✓ |

## Limitations

- Nested loops not fully supported
- Complex conditions may not parse correctly
- Non-linear updates limited
- Path explosion for deeply nested conditionals

## References

- Week 9: Boolean Combination of Invariants
- CAV03: Linear Invariant Generation
- CSUR14: Loop Invariants Survey
