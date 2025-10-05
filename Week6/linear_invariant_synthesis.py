import sys
import re
from dafny_parser import DafnyExtractor
from z3_solver import Z3Solver
from invariant_validator import InvariantValidator
from z3 import Int, And, Or, Implies, Not, sat

class LinearInvariantSynthesizer:
    """Synthesizes linear invariants for Dafny programs."""

    def __init__(self):
        self.parser = DafnyExtractor()
        self.solver = Z3Solver()
        self.validator = InvariantValidator()

    def synthesize(self, dafny_file_path: str):
        """Synthesizes linear invariants for the given Dafny program.

        Args:
            dafny_file_path: The path to the Dafny program.
        """
        parsed_data = self.parser.parse_file(dafny_file_path)
        if "error" in parsed_data:
            print(f"Error parsing Dafny file: {parsed_data['error']}")
            return

        for loop in parsed_data.get("loops", []):
            print(f"Synthesizing invariants for loop in {dafny_file_path}...")
            self._synthesize_for_loop(dafny_file_path, loop)

    def _synthesize_for_loop(self, dafny_file_path: str, loop: dict):
        """Synthesizes linear invariants for a single loop."""
        loop_body = loop.get("body", "")

        # Extract updates to variables from the loop body
        updates = self._extract_updates(loop_body)
        if not updates or len(updates) != 2:
            print("Skipping loop: Only supporting loops with 2 updated variables for now.")
            return

        x_var, y_var = list(updates.keys())[0], list(updates.keys())[1]

        # Generate constraints
        a, b, c = Int('a'), Int('b'), Int('c')
        
        # Inductive step constraint
        # a * f + b * g <= 0
        f = updates.get(x_var, 0)
        g = updates.get(y_var, 0)
        inductive_constraint = (a * f + b * g <= 0)

        # Base case constraint
        # c >= 0 (assuming initial values are 0)
        base_constraint = (c >= 0)

        # Non-triviality constraint
        non_trivial_constraint = Or(a != 0, b != 0)

        constraints = [inductive_constraint, base_constraint, non_trivial_constraint]

        # Solve constraints
        model = self.solver.solve(constraints)

        if model:
            # Construct invariant
            a_val = model.eval(a).as_long()
            b_val = model.eval(b).as_long()
            c_val = model.eval(c).as_long()
            invariant = f"{a_val} * {x_var} + {b_val} * {y_var} <= {c_val}"

            # Validate invariant
            print(f"Candidate invariant: {invariant}")
            if self.validator.validate(dafny_file_path, invariant):
                print(f"Successfully synthesized invariant: {invariant}")
            else:
                print("Failed to validate invariant.")
        else:
            print("Failed to find a solution for the constraints.")

    def _extract_updates(self, loop_body: str) -> dict:
        """Extracts the updates to the variables from the loop body."""
        updates = {}
        # Regex to find assignments of the form: var := var + const; or var := var - const;
        assignment_regex = re.compile(r'(\w+)\s*:=\s*\1\s*([+-])\s*(\d+);')
        matches = assignment_regex.findall(loop_body)

        for var, op, const in matches:
            value = int(const)
            if op == '-':
                value = -value
            updates[var] = value
        
        return updates

def main():
    if len(sys.argv) != 2:
        print("Usage: python linear_invariant_synthesis.py <dafny_file>")
        sys.exit(1)

    dafny_file = sys.argv[1]
    synthesizer = LinearInvariantSynthesizer()
    synthesizer.synthesize(dafny_file)

if __name__ == "__main__":
    main()
