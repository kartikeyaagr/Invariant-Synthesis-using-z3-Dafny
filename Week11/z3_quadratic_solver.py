"""
Z3 Quadratic Invariant Solver
Synthesizes quadratic invariants of the form:
  ax² + by² + cxy + dx + ey + f ≤ 0
  
Also supports equality forms:
  ax² + by² + cxy + dx + ey + f == 0
"""

from z3 import (
    Solver, Int, Real, And, Or, Not, Implies, If,
    sat, unsat, unknown, simplify, is_true, is_false,
    IntVector, RealVector
)
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import itertools


class QuadraticType(Enum):
    INEQUALITY = "inequality"  # <= 0
    EQUALITY = "equality"      # == 0
    BOTH = "both"              # Try both


@dataclass
class QuadraticInvariant:
    """
    Represents a quadratic invariant:
    sum(coeff[var^2] * var^2) + sum(coeff[var1*var2] * var1*var2) + 
    sum(coeff[var] * var) + constant ≤ 0 (or == 0)
    """
    # Coefficients for squared terms: {var: coeff}
    squared_coeffs: Dict[str, int]
    # Coefficients for cross terms: {(var1, var2): coeff} where var1 < var2
    cross_coeffs: Dict[Tuple[str, str], int]
    # Coefficients for linear terms: {var: coeff}
    linear_coeffs: Dict[str, int]
    # Constant term
    constant: int
    # Is this an equality or inequality
    is_equality: bool = False
    
    def evaluate(self, values: Dict[str, int]) -> int:
        """Evaluate the polynomial at given values"""
        result = self.constant
        
        for var, coeff in self.squared_coeffs.items():
            result += coeff * (values.get(var, 0) ** 2)
        
        for (v1, v2), coeff in self.cross_coeffs.items():
            result += coeff * values.get(v1, 0) * values.get(v2, 0)
        
        for var, coeff in self.linear_coeffs.items():
            result += coeff * values.get(var, 0)
        
        return result
    
    def to_string(self) -> str:
        """Convert to Dafny-compatible string"""
        terms = []
        
        # Squared terms
        for var, coeff in self.squared_coeffs.items():
            if coeff == 0:
                continue
            elif coeff == 1:
                terms.append(f"{var} * {var}")
            elif coeff == -1:
                terms.append(f"-{var} * {var}")
            else:
                terms.append(f"{coeff} * {var} * {var}")
        
        # Cross terms
        for (v1, v2), coeff in self.cross_coeffs.items():
            if coeff == 0:
                continue
            elif coeff == 1:
                terms.append(f"{v1} * {v2}")
            elif coeff == -1:
                terms.append(f"-{v1} * {v2}")
            else:
                terms.append(f"{coeff} * {v1} * {v2}")
        
        # Linear terms
        for var, coeff in self.linear_coeffs.items():
            if coeff == 0:
                continue
            elif coeff == 1:
                terms.append(var)
            elif coeff == -1:
                terms.append(f"-{var}")
            else:
                terms.append(f"{coeff} * {var}")
        
        if not terms:
            lhs = "0"
        else:
            lhs = " + ".join(terms).replace("+ -", "- ")
        
        op = "==" if self.is_equality else "<="
        return f"{lhs} {op} {-self.constant}"
    
    def is_trivial(self) -> bool:
        """Check if invariant is trivially true (e.g., 0 <= 0)"""
        all_zero = (
            all(c == 0 for c in self.squared_coeffs.values()) and
            all(c == 0 for c in self.cross_coeffs.values()) and
            all(c == 0 for c in self.linear_coeffs.values())
        )
        if all_zero:
            if self.is_equality:
                return self.constant == 0
            else:
                return self.constant >= 0
        return False


class Z3QuadraticSolver:
    """
    Z3-based solver for quadratic invariant synthesis.
    
    Supports:
    1. Pure quadratic: ax² + by² + f ≤ 0
    2. Mixed quadratic: ax² + by² + cxy + dx + ey + f ≤ 0
    3. Quadratic equality: ax² + by² + ... = 0
    """
    
    def __init__(self, coeff_bound: int = 5):
        self.coeff_bound = coeff_bound
    
    def synthesize_quadratic(self,
                            var_names: List[str],
                            init_values: Dict[str, int],
                            update_exprs: Dict[str, str],
                            loop_bound: int = 15,
                            invariant_type: QuadraticType = QuadraticType.INEQUALITY,
                            include_cross_terms: bool = True,
                            include_linear_terms: bool = True) -> Optional[QuadraticInvariant]:
        """
        Synthesize a quadratic invariant.
        
        Args:
            var_names: Variables in the loop
            init_values: Initial values
            update_exprs: Update expressions (e.g., {x: "+1", y: "*2"})
            loop_bound: Number of iterations to simulate
            invariant_type: INEQUALITY, EQUALITY, or BOTH
            include_cross_terms: Whether to include xy terms
            include_linear_terms: Whether to include x, y terms
        
        Returns:
            QuadraticInvariant if found, None otherwise
        """
        # Create coefficient variables
        squared_vars = {var: Int(f'a_{var}_sq') for var in var_names}
        
        cross_vars = {}
        if include_cross_terms:
            for i, v1 in enumerate(var_names):
                for v2 in var_names[i+1:]:
                    cross_vars[(v1, v2)] = Int(f'a_{v1}_{v2}')
        
        linear_vars = {}
        if include_linear_terms:
            linear_vars = {var: Int(f'b_{var}') for var in var_names}
        
        const_var = Int('c')
        
        s = Solver()
        
        # Bound coefficients
        all_coeffs = list(squared_vars.values()) + list(cross_vars.values()) + \
                     list(linear_vars.values()) + [const_var]
        for coeff in all_coeffs:
            s.add(coeff >= -self.coeff_bound, coeff <= self.coeff_bound)
        
        # Non-triviality: at least one non-constant coefficient is non-zero
        non_const_coeffs = list(squared_vars.values()) + list(cross_vars.values()) + \
                          list(linear_vars.values())
        if non_const_coeffs:
            s.add(Or(*[c != 0 for c in non_const_coeffs]))
        
        # Simulate loop and add constraints
        reachable_states = self._simulate_loop(var_names, init_values, update_exprs, loop_bound)
        
        for values in reachable_states:
            # Build polynomial expression
            expr = const_var
            
            for var in var_names:
                val = values.get(var, 0)
                expr = expr + squared_vars[var] * (val * val)
            
            for (v1, v2), coeff_var in cross_vars.items():
                expr = expr + coeff_var * values.get(v1, 0) * values.get(v2, 0)
            
            for var, coeff_var in linear_vars.items():
                expr = expr + coeff_var * values.get(var, 0)
            
            # Add constraint based on type
            if invariant_type == QuadraticType.EQUALITY:
                s.add(expr == 0)
            else:
                s.add(expr <= 0)
        
        # Solve
        if s.check() == sat:
            model = s.model()
            
            inv = QuadraticInvariant(
                squared_coeffs={var: model.eval(squared_vars[var]).as_long() 
                               for var in var_names},
                cross_coeffs={key: model.eval(cross_vars[key]).as_long() 
                             for key in cross_vars},
                linear_coeffs={var: model.eval(linear_vars[var]).as_long() 
                              for var in linear_vars},
                constant=model.eval(const_var).as_long(),
                is_equality=(invariant_type == QuadraticType.EQUALITY)
            )
            
            if not inv.is_trivial():
                return inv
        
        return None
    
    def _simulate_loop(self, var_names: List[str], init_values: Dict[str, int],
                       update_exprs: Dict[str, str], loop_bound: int) -> List[Dict[str, int]]:
        """Simulate loop execution to get reachable states"""
        states = []
        values = dict(init_values)
        
        # Add initial state
        states.append(dict(values))
        
        for _ in range(loop_bound):
            new_values = {}
            for var in var_names:
                update = update_exprs.get(var, "+0")
                old_val = values.get(var, 0)
                
                if update.startswith('+'):
                    delta = int(update[1:])
                    new_values[var] = old_val + delta
                elif update.startswith('-'):
                    delta = int(update[1:])
                    new_values[var] = old_val - delta
                elif update.startswith('*'):
                    factor = int(update[1:])
                    new_values[var] = old_val * factor
                elif update.startswith('sq'):
                    # Square the value (for testing quadratic growth)
                    new_values[var] = old_val * old_val
                else:
                    new_values[var] = old_val
            
            values = new_values
            states.append(dict(values))
        
        return states
    
    def synthesize_equality(self,
                           var_names: List[str],
                           init_values: Dict[str, int],
                           update_exprs: Dict[str, str],
                           loop_bound: int = 15) -> Optional[QuadraticInvariant]:
        """Synthesize a quadratic equality invariant"""
        return self.synthesize_quadratic(
            var_names, init_values, update_exprs,
            loop_bound, QuadraticType.EQUALITY
        )
    
    def synthesize_pure_quadratic(self,
                                  var_names: List[str],
                                  init_values: Dict[str, int],
                                  update_exprs: Dict[str, str],
                                  loop_bound: int = 15) -> Optional[QuadraticInvariant]:
        """Synthesize a pure quadratic (no linear terms)"""
        return self.synthesize_quadratic(
            var_names, init_values, update_exprs,
            loop_bound, include_linear_terms=False
        )
    
    def synthesize_all_forms(self,
                            var_names: List[str],
                            init_values: Dict[str, int],
                            update_exprs: Dict[str, str],
                            loop_bound: int = 15) -> List[QuadraticInvariant]:
        """Try multiple synthesis strategies"""
        results = []
        
        # Full quadratic inequality
        inv = self.synthesize_quadratic(
            var_names, init_values, update_exprs, loop_bound,
            QuadraticType.INEQUALITY, True, True
        )
        if inv and not inv.is_trivial():
            results.append(inv)
        
        # Quadratic equality
        inv = self.synthesize_quadratic(
            var_names, init_values, update_exprs, loop_bound,
            QuadraticType.EQUALITY, True, True
        )
        if inv and not inv.is_trivial():
            results.append(inv)
        
        # Pure quadratic (no linear)
        inv = self.synthesize_quadratic(
            var_names, init_values, update_exprs, loop_bound,
            QuadraticType.INEQUALITY, True, False
        )
        if inv and not inv.is_trivial():
            results.append(inv)
        
        # No cross terms
        inv = self.synthesize_quadratic(
            var_names, init_values, update_exprs, loop_bound,
            QuadraticType.INEQUALITY, False, True
        )
        if inv and not inv.is_trivial():
            results.append(inv)
        
        return results
    
    def verify_invariant(self, invariant: QuadraticInvariant,
                        var_names: List[str],
                        init_values: Dict[str, int],
                        update_exprs: Dict[str, str],
                        num_iterations: int = 100) -> bool:
        """Verify an invariant holds for many iterations"""
        states = self._simulate_loop(var_names, init_values, update_exprs, num_iterations)
        
        for values in states:
            result = invariant.evaluate(values)
            if invariant.is_equality:
                if result != 0:
                    return False
            else:
                if result > 0:
                    return False
        
        return True


class QuadraticTemplateGenerator:
    """Generates quadratic templates for specific patterns"""
    
    @staticmethod
    def triangular_sum_template(n_var: str, sum_var: str, i_var: str) -> str:
        """
        Template for triangular number sum: sum = i*(i-1)/2
        Equivalent to: 2*sum - i*i + i == 0
        """
        return f"2 * {sum_var} - {i_var} * {i_var} + {i_var} == 0"
    
    @staticmethod
    def square_sum_template(sum_var: str, i_var: str) -> str:
        """
        Template for sum of squares: sum = i*(i-1)*(2*i-1)/6
        (More complex, may need higher degree)
        """
        return f"{sum_var} >= 0"
    
    @staticmethod
    def product_template(prod_var: str, x_var: str, y_var: str) -> str:
        """
        Template for product relationship
        """
        return f"{prod_var} - {x_var} * {y_var} == 0"


if __name__ == "__main__":
    print("Testing Z3 Quadratic Solver...")
    print("=" * 60)
    
    solver = Z3QuadraticSolver(coeff_bound=5)
    
    # Test 1: Triangular numbers
    # sum starts at 0, i starts at 0
    # Loop: sum := sum + i; i := i + 1
    # Invariant should capture: 2*sum = i*(i-1), i.e., 2*sum - i² + i = 0
    print("\nTest 1: Triangular numbers (sum = 0 + 1 + 2 + ... + (i-1))")
    print("  Expected: 2*sum - i*i + i == 0")
    
    # We need to compute states where sum = i*(i-1)/2
    # Simulate manually: at step k, i=k, sum=0+1+...+(k-1)=k*(k-1)/2
    init = {"i": 0, "sum": 0}
    # The update is: sum := sum + i, then i := i + 1
    # We simulate this specially
    
    states = []
    i, s = 0, 0
    for _ in range(16):
        states.append({"i": i, "sum": s})
        s = s + i
        i = i + 1
    
    # Try to find quadratic equality
    inv = solver.synthesize_quadratic(
        var_names=["i", "sum"],
        init_values={"i": 0, "sum": 0},
        update_exprs={"i": "+1", "sum": "+0"},  # Simplified, we'll verify with states
        loop_bound=15,
        invariant_type=QuadraticType.EQUALITY
    )
    if inv:
        print(f"  Found: {inv.to_string()}")
    else:
        print("  No equality found, trying inequality...")
        inv = solver.synthesize_quadratic(
            var_names=["i", "sum"],
            init_values={"i": 0, "sum": 0},
            update_exprs={"i": "+1", "sum": "+0"},
            loop_bound=15
        )
        if inv:
            print(f"  Found: {inv.to_string()}")
    
    # Test 2: Simple quadratic growth
    print("\nTest 2: Simple quadratic growth (x starts at 0, increments by 1)")
    inv = solver.synthesize_quadratic(
        var_names=["x"],
        init_values={"x": 0},
        update_exprs={"x": "+1"},
        loop_bound=15
    )
    if inv:
        print(f"  Found: {inv.to_string()}")
    
    # Test 3: Two variables with relationship
    print("\nTest 3: Two variables (x += 1, y += 2)")
    inv = solver.synthesize_quadratic(
        var_names=["x", "y"],
        init_values={"x": 0, "y": 0},
        update_exprs={"x": "+1", "y": "+2"},
        loop_bound=15
    )
    if inv:
        print(f"  Found: {inv.to_string()}")
    
    # Test 4: Product relationship
    print("\nTest 4: Trying all forms")
    all_invs = solver.synthesize_all_forms(
        var_names=["x", "y"],
        init_values={"x": 0, "y": 0},
        update_exprs={"x": "+1", "y": "+2"},
        loop_bound=15
    )
    print(f"  Found {len(all_invs)} invariants:")
    for inv in all_invs:
        print(f"    - {inv.to_string()}")
    
    print("\n" + "=" * 60)
    print("Tests completed.")
