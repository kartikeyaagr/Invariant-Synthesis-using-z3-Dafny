"""
Z3 Solver Wrapper for Boolean Invariant Synthesis
Supports conjunctions and disjunctions of linear invariants.
"""

from z3 import (
    Solver, Int, Bool, And, Or, Not, Implies, 
    sat, unsat, unknown, simplify, is_true, is_false
)
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import itertools


class BoolOp(Enum):
    AND = "and"
    OR = "or"
    IMPLIES = "implies"


@dataclass
class LinearConstraint:
    """Represents ax + by + c <= 0 or ax + by + c == 0"""
    coefficients: Dict[str, int]  # variable -> coefficient
    constant: int
    is_equality: bool = False
    
    def to_string(self, var_names: List[str] = None) -> str:
        """Convert to readable string"""
        terms = []
        for var, coeff in self.coefficients.items():
            if coeff == 0:
                continue
            elif coeff == 1:
                terms.append(var)
            elif coeff == -1:
                terms.append(f"-{var}")
            else:
                terms.append(f"{coeff} * {var}")
        
        if not terms:
            terms = ["0"]
        
        lhs = " + ".join(terms).replace("+ -", "- ")
        op = "==" if self.is_equality else "<="
        return f"{lhs} {op} {-self.constant}"


@dataclass
class BooleanInvariant:
    """Represents a boolean combination of linear constraints"""
    constraints: List[LinearConstraint]
    operators: List[BoolOp]  # operators between constraints
    
    def to_string(self) -> str:
        """Convert to Dafny syntax"""
        if not self.constraints:
            return "true"
        
        if len(self.constraints) == 1:
            return self.constraints[0].to_string()
        
        result = self.constraints[0].to_string()
        for i, (op, constraint) in enumerate(zip(self.operators, self.constraints[1:])):
            c_str = constraint.to_string()
            if op == BoolOp.AND:
                result = f"({result}) && ({c_str})"
            elif op == BoolOp.OR:
                result = f"({result}) || ({c_str})"
            else:
                result = f"({result}) ==> ({c_str})"
        
        return result


class Z3BooleanSolver:
    """Z3-based solver for boolean combinations of linear invariants"""
    
    def __init__(self, coeff_bound: int = 10):
        self.coeff_bound = coeff_bound
        self.solver = Solver()
    
    def create_z3_vars(self, var_names: List[str]) -> Dict[str, Any]:
        """Create Z3 integer variables"""
        return {name: Int(name) for name in var_names}
    
    def linear_constraint_to_z3(self, constraint: LinearConstraint, 
                                 z3_vars: Dict[str, Any]) -> Any:
        """Convert LinearConstraint to Z3 expression"""
        expr = constraint.constant
        for var, coeff in constraint.coefficients.items():
            if var in z3_vars:
                expr = expr + coeff * z3_vars[var]
        
        if constraint.is_equality:
            return expr == 0
        else:
            return expr <= 0
    
    def boolean_invariant_to_z3(self, invariant: BooleanInvariant,
                                 z3_vars: Dict[str, Any]) -> Any:
        """Convert BooleanInvariant to Z3 expression"""
        if not invariant.constraints:
            return True
        
        z3_constraints = [self.linear_constraint_to_z3(c, z3_vars) 
                         for c in invariant.constraints]
        
        if len(z3_constraints) == 1:
            return z3_constraints[0]
        
        result = z3_constraints[0]
        for op, z3_c in zip(invariant.operators, z3_constraints[1:]):
            if op == BoolOp.AND:
                result = And(result, z3_c)
            elif op == BoolOp.OR:
                result = Or(result, z3_c)
            else:
                result = Implies(result, z3_c)
        
        return result

    def solve_for_coefficients(self, var_names: List[str], 
                               init_values: Dict[str, int],
                               update_exprs: Dict[str, str],
                               loop_bound: int = 20,
                               num_constraints: int = 1,
                               combination_type: BoolOp = BoolOp.AND) -> Optional[BooleanInvariant]:
        """
        Synthesize boolean combination of linear invariants.
        
        Args:
            var_names: Names of loop variables
            init_values: Initial values of variables
            update_exprs: Update expressions (e.g., {x: "+1", y: "+2"})
            loop_bound: Number of iterations to check
            num_constraints: Number of linear constraints to combine
            combination_type: How to combine constraints (AND/OR)
        """
        # Create coefficient variables for each constraint
        coeff_vars = []
        for i in range(num_constraints):
            coeffs = {}
            for var in var_names:
                coeffs[var] = Int(f'a_{i}_{var}')
            coeffs['const'] = Int(f'c_{i}')
            coeff_vars.append(coeffs)
        
        s = Solver()
        
        # Bound coefficients
        for coeffs in coeff_vars:
            for v in coeffs.values():
                s.add(v >= -self.coeff_bound, v <= self.coeff_bound)
        
        # Non-triviality: at least one coefficient is non-zero per constraint
        for coeffs in coeff_vars:
            var_coeffs = [coeffs[v] for v in var_names]
            s.add(Or(*[c != 0 for c in var_coeffs]))
        
        # Simulate loop execution and check invariant holds at each step
        for step in range(loop_bound + 1):
            # Compute variable values at this step
            values = {}
            for var in var_names:
                if step == 0:
                    values[var] = init_values.get(var, 0)
                else:
                    prev_val = init_values.get(var, 0)
                    update = update_exprs.get(var, "+0")
                    # Parse update expression
                    if update.startswith('+'):
                        delta = int(update[1:])
                        values[var] = prev_val + step * delta
                    elif update.startswith('-'):
                        delta = int(update[1:])
                        values[var] = prev_val - step * delta
                    elif update.startswith('*'):
                        factor = int(update[1:])
                        values[var] = prev_val * (factor ** step)
                    else:
                        values[var] = prev_val
            
            # Build constraint expressions for this step
            constraint_exprs = []
            for coeffs in coeff_vars:
                expr = coeffs['const']
                for var in var_names:
                    expr = expr + coeffs[var] * values[var]
                constraint_exprs.append(expr <= 0)
            
            # Combine constraints based on combination type
            if combination_type == BoolOp.AND:
                combined = And(*constraint_exprs) if len(constraint_exprs) > 1 else constraint_exprs[0]
            else:  # OR
                combined = Or(*constraint_exprs) if len(constraint_exprs) > 1 else constraint_exprs[0]
            
            s.add(combined)
        
        # Solve
        if s.check() == sat:
            model = s.model()
            constraints = []
            
            for coeffs in coeff_vars:
                lc = LinearConstraint(
                    coefficients={var: model.eval(coeffs[var]).as_long() 
                                 for var in var_names},
                    constant=model.eval(coeffs['const']).as_long()
                )
                constraints.append(lc)
            
            operators = [combination_type] * (num_constraints - 1)
            return BooleanInvariant(constraints=constraints, operators=operators)
        
        return None

    def synthesize_all_combinations(self, var_names: List[str],
                                    init_values: Dict[str, int],
                                    update_exprs: Dict[str, str],
                                    max_constraints: int = 3) -> List[BooleanInvariant]:
        """
        Synthesize multiple boolean combinations of invariants.
        Tries different numbers of constraints and combination types.
        """
        results = []
        
        # Try single constraints first
        inv = self.solve_for_coefficients(
            var_names, init_values, update_exprs,
            num_constraints=1
        )
        if inv:
            results.append(inv)
        
        # Try conjunctions
        for n in range(2, max_constraints + 1):
            inv = self.solve_for_coefficients(
                var_names, init_values, update_exprs,
                num_constraints=n, combination_type=BoolOp.AND
            )
            if inv:
                results.append(inv)
        
        # Try disjunctions
        for n in range(2, max_constraints + 1):
            inv = self.solve_for_coefficients(
                var_names, init_values, update_exprs,
                num_constraints=n, combination_type=BoolOp.OR
            )
            if inv:
                results.append(inv)
        
        return results

    def check_invariant_validity(self, invariant: BooleanInvariant,
                                  var_names: List[str],
                                  init_values: Dict[str, int],
                                  update_exprs: Dict[str, str],
                                  preconditions: List[str] = None) -> Tuple[bool, bool, bool]:
        """
        Check if invariant satisfies:
        1. Initialization: precondition => invariant at init
        2. Preservation: invariant && guard => invariant' (after update)
        3. Non-triviality: invariant is not always true
        
        Returns: (init_ok, preserve_ok, nontrivial)
        """
        z3_vars = self.create_z3_vars(var_names)
        z3_vars_prime = {f"{v}'": Int(f"{v}_prime") for v in var_names}
        
        inv_z3 = self.boolean_invariant_to_z3(invariant, z3_vars)
        
        # Check initialization
        s = Solver()
        init_state = And(*[z3_vars[v] == init_values.get(v, 0) for v in var_names])
        s.add(init_state)
        s.add(Not(inv_z3))
        init_ok = s.check() == unsat
        
        # Check preservation (simplified - assumes linear updates)
        s2 = Solver()
        s2.add(inv_z3)
        # Add update constraints
        for var in var_names:
            update = update_exprs.get(var, "+0")
            if update.startswith('+'):
                delta = int(update[1:])
                s2.add(z3_vars_prime[f"{var}'"] == z3_vars[var] + delta)
            elif update.startswith('-'):
                delta = int(update[1:])
                s2.add(z3_vars_prime[f"{var}'"] == z3_vars[var] - delta)
            else:
                s2.add(z3_vars_prime[f"{var}'"] == z3_vars[var])
        
        # Check that invariant holds for primed variables
        inv_prime = self.boolean_invariant_to_z3(
            invariant, 
            {v: z3_vars_prime[f"{v}'"] for v in var_names}
        )
        s2.add(Not(inv_prime))
        preserve_ok = s2.check() == unsat
        
        # Check non-triviality
        s3 = Solver()
        s3.add(Not(inv_z3))
        nontrivial = s3.check() == sat
        
        return init_ok, preserve_ok, nontrivial


def solve_constraints(constraints: List[Any]) -> Optional[Any]:
    """Simple wrapper for solving Z3 constraints"""
    s = Solver()
    s.add(constraints)
    if s.check() == sat:
        return s.model()
    return None


if __name__ == "__main__":
    # Test the solver
    solver = Z3BooleanSolver(coeff_bound=10)
    
    # Test case: x starts at 0, y starts at 0
    # Loop body: x := x + 1; y := y + 2
    var_names = ['x', 'y']
    init_values = {'x': 0, 'y': 0}
    update_exprs = {'x': '+1', 'y': '+2'}
    
    print("Testing boolean invariant synthesis...")
    print(f"Variables: {var_names}")
    print(f"Initial values: {init_values}")
    print(f"Updates: {update_exprs}")
    print()
    
    # Synthesize single constraint
    inv = solver.solve_for_coefficients(
        var_names, init_values, update_exprs,
        num_constraints=1
    )
    if inv:
        print(f"Single constraint: {inv.to_string()}")
    
    # Synthesize conjunction
    inv_conj = solver.solve_for_coefficients(
        var_names, init_values, update_exprs,
        num_constraints=2, combination_type=BoolOp.AND
    )
    if inv_conj:
        print(f"Conjunction: {inv_conj.to_string()}")
    
    # Synthesize disjunction
    inv_disj = solver.solve_for_coefficients(
        var_names, init_values, update_exprs,
        num_constraints=2, combination_type=BoolOp.OR
    )
    if inv_disj:
        print(f"Disjunction: {inv_disj.to_string()}")
    
    # Get all combinations
    print("\nAll synthesized invariants:")
    all_invs = solver.synthesize_all_combinations(
        var_names, init_values, update_exprs, max_constraints=3
    )
    for i, inv in enumerate(all_invs):
        print(f"  {i+1}. {inv.to_string()}")
