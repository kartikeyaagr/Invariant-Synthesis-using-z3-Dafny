# Week 12-13: Array Invariant Synthesis

This tool extends the invariant synthesis framework to handle **arrays and lists**, generating quantified invariants for array-manipulating loops.

## Features

- **Array Bounds Invariants**: `0 <= i <= a.Length`
- **Quantified Invariants**: `forall k :: 0 <= k < i ==> P(a[k])`
- **Pattern Detection**: Automatically identifies common array patterns
- **Multiple Array Support**: Handles programs with multiple arrays

## Supported Patterns

| Pattern | Description | Example Invariant |
|---------|-------------|-------------------|
| INIT | Array initialization | `forall k :: 0 <= k < i ==> a[k] == 0` |
| COPY | Copy src to dst | `forall k :: 0 <= k < i ==> dst[k] == src[k]` |
| ACCUMULATE | Sum/product over array | `sum == Sum(a[..i])` |
| TRANSFORM | In-place modification | `forall k :: 0 <= k < i ==> a[k] >= 0` |
| LINEAR_SCAN | Read-only traversal | `0 <= i <= a.Length` |
| SEARCH | Find element | `forall k :: 0 <= k < i ==> a[k] != key` |

## Installation

```bash
pip install z3-solver
```

## Usage

### Basic Usage

```bash
# Synthesize array invariants
python main.py benchmarks/bench1_init.dfy

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
-c, --coeff-bound N   Coefficient bound (default: 10)
-o, --output FILE     Output file with inserted invariants
--json                Output as JSON
--test                Run test suite
--benchmark           Run benchmarks
```

## Benchmarks

| Benchmark | Pattern | Description |
|-----------|---------|-------------|
| bench1_init | INIT | Initialize array to zeros |
| bench2_sum | ACCUMULATE | Compute array sum |
| bench3_copy | COPY | Copy array elements |
| bench4_max | LINEAR_SCAN | Find maximum element |
| bench5_search | SEARCH | Linear search |
| bench6_reverse | TRANSFORM | Reverse array in place |
| bench7_count | ACCUMULATE | Count occurrences |
| bench8_all_positive | LINEAR_SCAN | Check all positive |

## Invariant Types Generated

### 1. Bounds Invariants
```dafny
invariant 0 <= i <= a.Length
```

### 2. Processed Elements (forall)
```dafny
invariant forall k :: 0 <= k < i ==> a[k] == 0
```

### 3. Unprocessed Elements Unchanged
```dafny
invariant forall k :: i <= k < a.Length ==> a[k] == old(a[k])
```

### 4. Copy Invariants
```dafny
invariant forall k :: 0 <= k < i ==> dst[k] == src[k]
```

### 5. Search Invariants
```dafny
invariant forall k :: 0 <= k < i ==> a[k] != key
```

### 6. Accumulator Invariants
```dafny
invariant sum == Sum(a[..i])
invariant count >= 0
```

### 7. Sorted Invariants
```dafny
invariant forall j, k :: 0 <= j < k < i ==> a[j] <= a[k]
```

## Architecture

```
week12/
├── main.py                 # Entry point
├── array_synthesis.py      # Main synthesis module
├── array_analyzer.py       # Array pattern analysis
├── z3_array_solver.py      # Z3 solver for arrays
├── benchmarks/             # 8 test programs
│   ├── bench1_init.dfy
│   ├── bench2_sum.dfy
│   ├── bench3_copy.dfy
│   ├── bench4_max.dfy
│   ├── bench5_search.dfy
│   ├── bench6_reverse.dfy
│   ├── bench7_count.dfy
│   └── bench8_all_positive.dfy
├── tests/
│   └── test_arrays.py
└── README.md
```

## Algorithm

### Phase 1: Array Analysis

1. Parse Dafny source to find array variables
2. Extract loop structure and array accesses
3. Identify read vs. write operations
4. Determine loop index and direction

### Phase 2: Pattern Detection

1. Classify loop based on array operations:
   - Write-only → INIT or TRANSFORM
   - Read-only → LINEAR_SCAN or SEARCH
   - Both → COPY or TRANSFORM
   - With accumulator → ACCUMULATE

### Phase 3: Template Instantiation

1. Generate bounds invariants
2. Instantiate pattern-specific quantified templates
3. Generate relationship invariants between variables

### Phase 4: Confidence Ranking

Sort invariants by confidence:
- Bounds: 1.0 (always needed)
- Pattern-specific quantified: 0.75-0.9
- Generic properties: 0.5-0.7
- Relationship guesses: 0.3

## API Example

```python
from array_synthesis import ArrayInvariantSynthesizer

synthesizer = ArrayInvariantSynthesizer()

# From specification
invariants = synthesizer.synthesize_from_spec(
    loop_condition="i < a.Length",
    loop_body="{ a[i] := 0; i := i + 1; }",
    variables=["i", "a"],
    arrays=["a"]
)

for inv in invariants:
    print(inv)
# Output:
#   0 <= i
#   i <= a.Length
#   forall k :: 0 <= k < i ==> a[k] == 0
```

## Comparison with Previous Weeks

| Feature | Week 6-11 | Week 12-13 |
|---------|-----------|------------|
| Scalar variables | ✓ | ✓ |
| Linear invariants | ✓ | ✓ |
| Quadratic | ✓ (Week 11) | ✓ |
| Boolean combinations | ✓ (Week 9) | ✓ |
| Arrays | ✗ | ✓ |
| Quantified (forall/exists) | ✗ | ✓ |
| Pattern detection | Limited | Full |

## Limitations

- Nested loops have limited support
- Complex index expressions may not be detected
- Non-integer arrays not fully supported
- Ghost functions for sum/product need manual definition

## Theoretical Background

Array invariant synthesis uses:
- **Template-based synthesis**: Predefined patterns for common operations
- **Pattern matching**: Classify loop behavior from syntactic analysis
- **Quantifier instantiation**: Generate forall/exists invariants

Unlike scalar synthesis, array invariants typically require:
- Quantified properties over ranges
- Frame conditions (what's unchanged)
- Multiset equality for permutation properties

## Future Extensions

- Nested array access (matrices)
- Sequence operations (append, slice)
- Set and map data structures
- Recursive function support
- Stronger Z3 verification

## References

- Weeks 6-11: Scalar invariant synthesis
- Dafny documentation: Arrays and sequences
- Furia et al.: Loop invariants (CSUR'14)
