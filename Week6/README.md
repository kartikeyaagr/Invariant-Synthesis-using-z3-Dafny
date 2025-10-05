# Linear Invariant Synthesis Tool

This tool automatically generates linear invariants of the form `ax + by <= c` for Dafny programs.

## Usage

To use the tool, run the `linear_invariant_synthesis.py` script from the command line, passing the path to a Dafny file as an argument:

```bash
python3 linear_invariant_synthesis.py <path_to_dafny_file>
```

## Example

To synthesize an invariant for the `test_program.dfy` file, run the following command:

```bash
python3 linear_invariant_synthesis.py test_program.dfy
```

This will output the synthesized invariant:

```
Synthesizing invariants for loop in test_program.dfy...
Candidate invariant: -2 * x + 1 * y <= 0
Successfully synthesized invariant: -2 * x + 1 * y <= 0
```

## Dependencies

- Python 3
- Z3 Solver (`z3-solver` pip package)
- Dafny
