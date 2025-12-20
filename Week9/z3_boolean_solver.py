"""
Z3 Solver Wrapper for Boolean Invariant Synthesis
Supports conjunctions and disjunctions of linear invariants.

UPDATED: Now includes equality synthesis, diverse solutions, and better filtering.
"""

from z3 import (
    Solver, Int, Bool, And, Or, Not, Implies, 
    sat, unsat, unknown, simplify, is_true, is_false
)
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
from math import gcd
from functools import reduce


class BoolOp(Enum):
    AND = "and"
    OR = "or"
    IMPLIES = "implies"


class ConstraintType(Enum):
    """Type of constraint to synthesize"""
    INEQUALITY = "inequality"  # <= 0
    EQUALITY = "equality"      # == 0


@dataclass
class LinearConstraint:
    """Represents ax + by + c <= 0 or ax + by + c == 0"""
    coefficients: Dict[str, int]  # variable -> coefficient
    constant: int
    is_equality: bool = False
    
    def to_string(self, var_names: List[str] = None) -> str:
        """Convert to readable Dafny syntax"""
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
    
    def is_trivial(self) -> bool:
        """Check if this constraint is trivially true or useless"""
        # All variable coefficients are zero
        if all(c == 0 for c in self.coefficients.values()):
            if self.is_equality:
                return self.constant == 0  # 0 == 0 is trivial
            else:
                return self.constant >= 0  # 0 <= positive is trivial
        return False
    
    def signature(self) -> tuple:
        """Return a hashable signature for deduplication"""
        coeffs = list(self.coefficients.values()) + [self.constant]
        non_zero = [abs(c) for c in coeffs if c != 0]
        
        if not non_zero:
            return (tuple(), 0, self.is_equality)
        
        g = reduce(gcd, non_zero)
        
        # Normalize coefficients
        norm_coeffs = {k: v // g for k, v in self.coefficients.items()}
        norm_const = self.constant // g
        
        # Make first non-zero coefficient positive for consistency
        first_nonzero = None
        for var in sorted(norm_coeffs.keys()):
            if norm_coeffs[var] != 0:
                first_nonzero = norm_coeffs[var]
                break
        
        if first_nonzero and first_nonzero < 0:
            norm_coeffs = {k: -v for k, v in norm_coeffs.items()}
            norm_const = -norm_const
        
        items = tuple(sorted(norm_coeffs.items()))
        return (items, norm_const, self.is_equality)


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
    
    def is_trivial(self) -> bool:
        """Check if all constraints are trivial"""
        return all(c.is_trivial() for c in self.constraints)
    
    def signature(self) -> tuple:
        """Return a hashable signature for deduplication"""
        return tuple(c.signature() for c in self.constraints)


class Z3BooleanSolver:
    """Z3-based solver for boolean combinations of linear invariants"""
    
    def __init__(self, coeff_bound: int = 10):
        self.coeff_bound = coeff_bound
    
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
                               combination_type: BoolOp = BoolOp.AND,
                               constraint_type: ConstraintType = ConstraintType.INEQUALITY,
                               blocked_solutions: List[Dict] = None) -> Optional[BooleanInvariant]:
        """
        Synthesize boolean combination of linear invariants.
        
        Args:
            var_names: Names of loop variables
            init_values: Initial values of variables
            update_exprs: Update expressions (e.g., {x: "+1", y: "+2"})
            loop_bound: Number of iterations to check
            num_constraints: Number of linear constraints to combine
            combination_type: How to combine constraints (AND/OR)
            constraint_type: INEQUALITY (<=) or EQUALITY (==)
            blocked_solutions: Previous solutions to avoid (for diverse synthesis)
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
        
        # Non-triviality: at least one variable coefficient is non-zero per constraint
        for coeffs in coeff_vars:
            var_coeffs = [coeffs[v] for v in var_names]
            s.add(Or(*[c != 0 for c in var_coeffs]))
        
        # Block previous solutions if provided
        if blocked_solutions:
            for blocked in blocked_solutions:
                block_clause = []
                for i, coeffs in enumerate(coeff_vars):
                    for var in var_names:
                        key = f'a_{i}_{var}'
                        if key in blocked:
                            block_clause.append(coeffs[var] != blocked[key])
                    key = f'c_{i}'
                    if key in blocked:
                        block_clause.append(coeffs['const'] != blocked[key])
                if block_clause:
                    s.add(Or(*block_clause))
        
        # Simulate loop execution and check invariant holds at each step
        for step in range(loop_bound + 1):
            # Compute variable values at this step
            values = {}
            for var in var_names:
                base_val = init_values.get(var, 0)
                update = update_exprs.get(var, "+0")
                
                if step == 0:
                    values[var] = base_val
                else:
                    # Parse update expression
                    if update.startswith('+'):
                        try:
                            delta = int(update[1:])
                            values[var] = base_val + step * delta
                        except ValueError:
                            values[var] = base_val
                    elif update.startswith('-'):
                        try:
                            delta = int(update[1:])
                            values[var] = base_val - step * delta
                        except ValueError:
                            values[var] = base_val
                    elif update.startswith('*'):
                        try:
                            factor = int(update[1:])
                            values[var] = base_val * (factor ** step)
                        except ValueError:
                            values[var] = base_val
                    else:
                        values[var] = base_val
            
            # Build constraint expressions for this step
            constraint_exprs = []
            for coeffs in coeff_vars:
                expr = coeffs['const']
                for var in var_names:
                    expr = expr + coeffs[var] * values[var]
                
                if constraint_type == ConstraintType.EQUALITY:
                    constraint_exprs.append(expr == 0)
                else:
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
                    constant=model.eval(coeffs['const']).as_long(),
                    is_equality=(constraint_type == ConstraintType.EQUALITY)
                )
                constraints.append(lc)
            
            operators = [combination_type] * (num_constraints - 1)
            inv = BooleanInvariant(constraints=constraints, operators=operators)
            
            # Skip trivial invariants by recursively blocking
            if inv.is_trivial():
                new_blocked = list(blocked_solutions) if blocked_solutions else []
                solution_dict = {}
                for i, coeffs in enumerate(coeff_vars):
                    for var in var_names:
                        solution_dict[f'a_{i}_{var}'] = model.eval(coeffs[var]).as_long()
                    solution_dict[f'c_{i}'] = model.eval(coeffs['const']).as_long()
                new_blocked.append(solution_dict)
                
                # Limit recursion depth
                if len(new_blocked) < 20:
                    return self.solve_for_coefficients(
                        var_names, init_values, update_exprs, loop_bound,
                        num_constraints, combination_type, constraint_type,
                        new_blocked
                    )
                return None
            
            return inv
        
        return None

    def synthesize_diverse(self, var_names: List[str],
                           init_values: Dict[str, int],
                           update_exprs: Dict[str, str],
                           num_solutions: int = 5,
                           constraint_type: ConstraintType = ConstraintType.INEQUALITY,
                           loop_bound: int = 20) -> List[BooleanInvariant]:
        """
        Synthesize multiple diverse single-constraint invariants.
        Each call blocks the previous solution to get different results.
        """
        results = []
        blocked = []
        seen_signatures = set()
        
        attempts = 0
        max_attempts = num_solutions * 3
        
        while len(results) < num_solutions and attempts < max_attempts:
            attempts += 1
            
            inv = self.solve_for_coefficients(
                var_names, init_values, update_exprs,
                loop_bound=loop_bound,
                num_constraints=1,
                constraint_type=constraint_type,
                blocked_solutions=blocked
            )
            
            if inv is None:
                break
            
            # Check if truly different using signature
            sig = inv.signature()
            if sig not in seen_signatures and not inv.is_trivial():
                seen_signatures.add(sig)
                results.append(inv)
            
            # Block this solution for next iteration
            solution_dict = {}
            for var in var_names:
                solution_dict[f'a_0_{var}'] = inv.constraints[0].coefficients[var]
            solution_dict['c_0'] = inv.constraints[0].constant
            blocked.append(solution_dict)
        
        return results

    def synthesize_bound_invariants(self, var_names: List[str],
                                     init_values: Dict[str, int],
                                     update_exprs: Dict[str, str],
                                     loop_bound: int = 20) -> List[BooleanInvariant]:
        """
        Synthesize bound invariants of the form:
        - x >= 0  (lower bounds)
        - x <= n  (upper bounds involving other variables)
        """
        results = []
        
        # For each variable that gets updated (not parameters)
        for var in var_names:
            update = update_exprs.get(var, "+0")
            init_val = init_values.get(var, 0)
            
            # Skip parameters (no update)
            if update == "+0" and var not in ['i', 'j', 'x', 'y', 'z', 'k', 'count', 'sum', 'result']:
                continue
            
            # If starts at 0 and increases, x >= 0 is an invariant
            if init_val == 0 and update.startswith('+'):
                # Create: -x <= 0 (i.e., x >= 0)
                inv = BooleanInvariant(
                    constraints=[LinearConstraint(
                        coefficients={v: -1 if v == var else 0 for v in var_names},
                        constant=0,
                        is_equality=False
                    )],
                    operators=[]
                )
                results.append(inv)
            
            # If starts at 0 and decreases, x <= 0 is an invariant
            if init_val == 0 and update.startswith('-'):
                inv = BooleanInvariant(
                    constraints=[LinearConstraint(
                        coefficients={v: 1 if v == var else 0 for v in var_names},
                        constant=0,
                        is_equality=False
                    )],
                    operators=[]
                )
                results.append(inv)
        
        # Try to find upper bounds involving parameters (e.g., i <= n)
        # This requires trying combinations with parameters
        for var in var_names:
            update = update_exprs.get(var, "+0")
            if update == "+0":
                continue  # This is likely a parameter
                
            for param in var_names:
                if update_exprs.get(param, "+0") != "+0":
                    continue  # Not a parameter
                
                # Try: var <= param (encoded as var - param <= 0)
                # Check if it holds through simulation
                holds = True
                for step in range(loop_bound + 1):
                    var_val = init_values.get(var, 0)
                    param_val = init_values.get(param, 0)
                    
                    upd = update_exprs.get(var, "+0")
                    if upd.startswith('+'):
                        try:
                            delta = int(upd[1:])
                            var_val = init_values.get(var, 0) + step * delta
                        except:
                            pass
                    
                    # For upper bound, we need var_val <= param_val
                    # But param doesn't change, so we check relative values
                    # This is tricky - we're simulating with param=0
                    # The invariant i <= n only holds if we assume the loop guard
                
                # For now, just add i <= n style invariants if i starts at 0 and increases
                if init_values.get(var, 0) == 0 and update.startswith('+'):
                    inv = BooleanInvariant(
                        constraints=[LinearConstraint(
                            coefficients={v: (1 if v == var else (-1 if v == param else 0)) for v in var_names},
                            constant=0,
                            is_equality=False
                        )],
                        operators=[]
                    )
                    results.append(inv)
        
        return results

    def synthesize_all_combinations(self, var_names: List[str],
                                    init_values: Dict[str, int],
                                    update_exprs: Dict[str, str],
                                    max_constraints: int = 3,
                                    loop_bound: int = 20) -> List[BooleanInvariant]:
        """
        Synthesize multiple boolean combinations of invariants.
        Now includes equalities, diverse solutions, and bound invariants.
        """
        results = []
        seen_signatures = set()
        
        def add_if_unique(inv: BooleanInvariant):
            if inv and not inv.is_trivial():
                sig = inv.signature()
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    results.append(inv)
        
        # 1. Try diverse single EQUALITIES first (most useful for relationships like j == 2*i)
        diverse_eq = self.synthesize_diverse(
            var_names, init_values, update_exprs,
            num_solutions=5,
            constraint_type=ConstraintType.EQUALITY,
            loop_bound=loop_bound
        )
        for inv in diverse_eq:
            add_if_unique(inv)
        
        # 2. Try diverse single INEQUALITIES
        diverse_ineq = self.synthesize_diverse(
            var_names, init_values, update_exprs,
            num_solutions=5,
            constraint_type=ConstraintType.INEQUALITY,
            loop_bound=loop_bound
        )
        for inv in diverse_ineq:
            add_if_unique(inv)
        
        # 3. Add common bound invariants (0 <= i, i <= n, etc.)
        bound_invs = self.synthesize_bound_invariants(
            var_names, init_values, update_exprs, loop_bound
        )
        for inv in bound_invs:
            add_if_unique(inv)
        
        # 4. Try conjunctions of inequalities
        for n in range(2, max_constraints + 1):
            inv = self.solve_for_coefficients(
                var_names, init_values, update_exprs,
                loop_bound=loop_bound,
                num_constraints=n, 
                combination_type=BoolOp.AND,
                constraint_type=ConstraintType.INEQUALITY
            )
            add_if_unique(inv)
        
        # 5. Try conjunctions of equalities
        for n in range(2, max_constraints + 1):
            inv = self.solve_for_coefficients(
                var_names, init_values, update_exprs,
                loop_bound=loop_bound,
                num_constraints=n,
                combination_type=BoolOp.AND,
                constraint_type=ConstraintType.EQUALITY
            )
            add_if_unique(inv)
        
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
        
        # Check preservation
        s2 = Solver()
        s2.add(inv_z3)
        for var in var_names:
            update = update_exprs.get(var, "+0")
            if update.startswith('+'):
                try:
                    delta = int(update[1:])
                    s2.add(z3_vars_prime[f"{var}'"] == z3_vars[var] + delta)
                except ValueError:
                    s2.add(z3_vars_prime[f"{var}'"] == z3_vars[var])
            elif update.startswith('-'):
                try:
                    delta = int(update[1:])
                    s2.add(z3_vars_prime[f"{var}'"] == z3_vars[var] - delta)
                except ValueError:
                    s2.add(z3_vars_prime[f"{var}'"] == z3_vars[var])
            else:
                s2.add(z3_vars_prime[f"{var}'"] == z3_vars[var])
        
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
    solver = Z3BooleanSolver(coeff_bound=10)
    
    print("=" * 60)
    print("Testing Boolean Invariant Synthesis")
    print("=" * 60)
    
    var_names = ['i', 'j', 'n']
    init_values = {'i': 0, 'j': 0, 'n': 0}
    update_exprs = {'i': '+1', 'j': '+2', 'n': '+0'}
    
    print(f"\nVariables: {var_names}")
    print(f"Initial values: {init_values}")
    print(f"Updates: {update_exprs}")
    
    print("\n--- Diverse Equalities ---")
    diverse_eq = solver.synthesize_diverse(
        var_names, init_values, update_exprs,
        num_solutions=5,
        constraint_type=ConstraintType.EQUALITY
    )
    for inv in diverse_eq:
        print(f"  {inv.to_string()}")
    
    print("\n--- Diverse Inequalities ---")
    diverse_ineq = solver.synthesize_diverse(
        var_names, init_values, update_exprs,
        num_solutions=5,
        constraint_type=ConstraintType.INEQUALITY
    )
    for inv in diverse_ineq:
        print(f"  {inv.to_string()}")
    
    print("\n--- All Synthesized Invariants ---")
    all_invs = solver.synthesize_all_combinations(
        var_names, init_values, update_exprs, max_constraints=2
    )
    for i, inv in enumerate(all_invs):
        eq_marker = "[EQ]" if inv.constraints[0].is_equality else "[LE]"
        print(f"  {i+1}. {eq_marker} {inv.to_string()}")
    
    print("\n" + "=" * 60)
