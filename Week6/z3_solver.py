from z3 import Solver, Int, sat, And

class Z3Solver:
    """A simple interface to the Z3 solver."""

    def solve(self, constraints):
        """Tries to solve the given constraints.

        Args:
            constraints: A list of Z3 constraints.

        Returns:
            A Z3 model if the constraints are satisfiable, None otherwise.
        """
        solver = Solver()
        solver.add(constraints)
        if solver.check() == sat:
            return solver.model()
        return None
