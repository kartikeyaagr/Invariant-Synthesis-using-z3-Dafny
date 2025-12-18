"""
Z3 Disjunctive Solver for Path-Sensitive Invariant Synthesis
Synthesizes disjunctive invariants of the form:
  (path1_cond && inv1) || (path2_cond && inv2) || ...
"""

from z3 import (
    Solver, Int, Bool, Real, And, Or, Not, Implies, If,
    sat, unsat, unknown, simplify, is_true, is_false,
    ForAll, Exists
)
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import itertools

from path_analyzer import ExecutionPath, LoopPathAnalysis, PathType


@dataclass
class DisjunctInvariant:
    """A single disjunct in a disjunctive invariant"""
    path_condition: Optional[str]  # Guard for this disjunct (None = always)
    linear_constraint: Dict[str, int]  # {var: coeff}
    constant: int
    
    def to_string(self, include_guard: bool = True) -> str:
        """Convert to Dafny-compatible string"""
        # Build linear expression
        terms = []
        for var, coeff in self.linear_constraint.items():
            if coeff == 0:
                continue
            elif coeff == 1:
                terms.append(var)
            elif coeff == -1:
                terms.append(f"-{var}")
            else:
                terms.append(f"{coeff} * {var}")
        
        if not terms:
            constraint_str = f"0 <= {-self.constant}"
        else:
            lhs = " + ".join(terms).replace("+ -", "- ")
            constraint_str = f"{lhs} <= {-self.constant}"
        
        if include_guard and self.path_condition:
            return f"({self.path_condition} ==> {constraint_str})"
        return constraint_str


@dataclass 
class DisjunctiveInvariant:
    """A full disjunctive invariant: D1 || D2 || ... || Dn"""
    disjuncts: List[DisjunctInvariant]
    is_path_sensitive: bool = False
    
    def to_string(self) -> str:
        """Convert to Dafny-compatible string"""
        if not self.disjuncts:
            return "true"
        
        if len(self.disjuncts) == 1:
            return self.disjuncts[0].to_string()
        
        parts = [d.to_string() for d in self.disjuncts]
        return " || ".join(f"({p})" for p in parts)
    
    def to_conjunctive_string(self) -> str:
        """Convert to conjunction form if path-sensitive"""
        if not self.is_path_sensitive:
            return self.to_string()
        
        # Path-sensitive: (guard1 ==> inv1) && (guard2 ==> inv2)
        parts = []
        for d in self.disjuncts:
            parts.append(d.to_string(include_guard=True))
        
        return " && ".join(parts)


class Z3DisjunctiveSolver:
    """
    Z3-based solver for disjunctive invariant synthesis.
    
    Handles:
    1. Simple disjunctions: inv1 || inv2
    2. Path-sensitive: (cond && inv1) || (!cond && inv2)
    3. Guarded disjunctions: (guard1 ==> inv1) && (guard2 ==> inv2)
    """
    
    def __init__(self, coeff_bound: int = 10):
        self.coeff_bound = coeff_bound
    
    def synthesize_disjunctive(self, 
                               var_names: List[str],
                               init_values: Dict[str, int],
                               path_analysis: LoopPathAnalysis,
                               num_disjuncts: int = 2,
                               loop_bound: int = 20) -> Optional[DisjunctiveInvariant]:
        """
        Synthesize a disjunctive invariant based on path analysis.
        
        Args:
            var_names: Variables in scope
            init_values: Initial values
            path_analysis: Analysis of execution paths
            num_disjuncts: Number of disjuncts to try
            loop_bound: Iterations to simulate
        
        Returns:
            DisjunctiveInvariant if found, None otherwise
        """
        # Create coefficient variables for each disjunct
        coeff_vars = []
        for d in range(num_disjuncts):
            coeffs = {var: Int(f'a_{d}_{var}') for var in var_names}
            coeffs['const'] = Int(f'c_{d}')
            coeff_vars.append(coeffs)
        
        s = Solver()
        
        # Bound coefficients
        for coeffs in coeff_vars:
            for v in coeffs.values():
                s.add(v >= -self.coeff_bound, v <= self.coeff_bound)
        
        # Non-triviality for each disjunct
        for coeffs in coeff_vars:
            var_coeffs = [coeffs[v] for v in var_names]
            s.add(Or(*[c != 0 for c in var_coeffs]))
        
        # Simulate execution along each path
        for path in path_analysis.paths:
            self._add_path_constraints(
                s, coeff_vars, var_names, init_values, 
                path, loop_bound
            )
        
        # Solve
        if s.check() == sat:
            model = s.model()
            disjuncts = []
            
            for d, coeffs in enumerate(coeff_vars):
                path_cond = None
                if d < len(path_analysis.paths):
                    path_cond = path_analysis.paths[d].get_path_condition()
                    if path_cond == "true":
                        path_cond = None
                
                disjunct = DisjunctInvariant(
                    path_condition=path_cond,
                    linear_constraint={
                        var: model.eval(coeffs[var]).as_long()
                        for var in var_names
                    },
                    constant=model.eval(coeffs['const']).as_long()
                )
                disjuncts.append(disjunct)
            
            return DisjunctiveInvariant(
                disjuncts=disjuncts,
                is_path_sensitive=len(path_analysis.paths) > 1
            )
        
        return None
    
    def _add_path_constraints(self, solver: Solver,
                              coeff_vars: List[Dict],
                              var_names: List[str],
                              init_values: Dict[str, int],
                              path: ExecutionPath,
                              loop_bound: int):
        """Add constraints for a specific execution path"""
        
        # Get update deltas for this path
        deltas = {}
        for var in var_names:
            if var in path.updates and path.updates[var].delta is not None:
                deltas[var] = path.updates[var].delta
            else:
                deltas[var] = 0
        
        # For each iteration along this path
        for step in range(loop_bound + 1):
            # Compute variable values at this step
            values = {}
            for var in var_names:
                base = init_values.get(var, 0)
                values[var] = base + step * deltas.get(var, 0)
            
            # At least one disjunct must hold
            disjunct_constraints = []
            for coeffs in coeff_vars:
                expr = coeffs['const']
                for var in var_names:
                    expr = expr + coeffs[var] * values[var]
                disjunct_constraints.append(expr <= 0)
            
            # Add: at least one disjunct satisfied
            solver.add(Or(*disjunct_constraints))
    
    def synthesize_path_sensitive(self,
                                  var_names: List[str],
                                  init_values: Dict[str, int],
                                  path_analysis: LoopPathAnalysis,
                                  loop_bound: int = 20) -> Optional[DisjunctiveInvariant]:
        """
        Synthesize path-sensitive invariants where each path gets its own invariant.
        
        Form: (path1_guard ==> inv1) && (path2_guard ==> inv2) && ...
        """
        num_paths = len(path_analysis.paths)
        if num_paths == 0:
            return None
        
        # One set of coefficients per path
        coeff_vars = []
        for p in range(num_paths):
            coeffs = {var: Int(f'a_{p}_{var}') for var in var_names}
            coeffs['const'] = Int(f'c_{p}')
            coeff_vars.append(coeffs)
        
        s = Solver()
        
        # Bound and non-triviality
        for coeffs in coeff_vars:
            for v in coeffs.values():
                s.add(v >= -self.coeff_bound, v <= self.coeff_bound)
            var_coeffs = [coeffs[v] for v in var_names]
            s.add(Or(*[c != 0 for c in var_coeffs]))
        
        # For each path, the corresponding invariant must hold along that path
        for p_idx, path in enumerate(path_analysis.paths):
            coeffs = coeff_vars[p_idx]
            
            # Get deltas for this path
            deltas = {}
            for var in var_names:
                if var in path.updates and path.updates[var].delta is not None:
                    deltas[var] = path.updates[var].delta
                else:
                    deltas[var] = 0
            
            # Invariant must hold at each step along this path
            for step in range(loop_bound + 1):
                values = {
                    var: init_values.get(var, 0) + step * deltas.get(var, 0)
                    for var in var_names
                }
                
                expr = coeffs['const']
                for var in var_names:
                    expr = expr + coeffs[var] * values[var]
                
                s.add(expr <= 0)
        
        # Solve
        if s.check() == sat:
            model = s.model()
            disjuncts = []
            
            for p_idx, path in enumerate(path_analysis.paths):
                coeffs = coeff_vars[p_idx]
                path_cond = path.get_path_condition()
                if path_cond == "true":
                    path_cond = None
                
                disjunct = DisjunctInvariant(
                    path_condition=path_cond,
                    linear_constraint={
                        var: model.eval(coeffs[var]).as_long()
                        for var in var_names
                    },
                    constant=model.eval(coeffs['const']).as_long()
                )
                disjuncts.append(disjunct)
            
            return DisjunctiveInvariant(
                disjuncts=disjuncts,
                is_path_sensitive=True
            )
        
        return None
    
    def synthesize_alternating(self,
                               var_names: List[str],
                               init_values: Dict[str, int],
                               even_deltas: Dict[str, int],
                               odd_deltas: Dict[str, int],
                               loop_bound: int = 20) -> Optional[DisjunctiveInvariant]:
        """
        Synthesize invariants for loops with alternating behavior.
        E.g., even iterations do one thing, odd iterations do another.
        
        Form: (i % 2 == 0 && inv_even) || (i % 2 == 1 && inv_odd)
        """
        # Coefficients for even and odd invariants
        even_coeffs = {var: Int(f'a_even_{var}') for var in var_names}
        even_coeffs['const'] = Int('c_even')
        
        odd_coeffs = {var: Int(f'a_odd_{var}') for var in var_names}
        odd_coeffs['const'] = Int('c_odd')
        
        s = Solver()
        
        # Bounds
        for coeffs in [even_coeffs, odd_coeffs]:
            for v in coeffs.values():
                s.add(v >= -self.coeff_bound, v <= self.coeff_bound)
            var_coeffs = [coeffs[v] for v in var_names]
            s.add(Or(*[c != 0 for c in var_coeffs]))
        
        # Simulate alternating execution
        values = dict(init_values)
        for step in range(loop_bound + 1):
            if step % 2 == 0:
                # Even iteration - use even invariant
                expr = even_coeffs['const']
                for var in var_names:
                    expr = expr + even_coeffs[var] * values.get(var, 0)
                s.add(expr <= 0)
                
                # Apply even deltas
                for var, delta in even_deltas.items():
                    values[var] = values.get(var, 0) + delta
            else:
                # Odd iteration - use odd invariant  
                expr = odd_coeffs['const']
                for var in var_names:
                    expr = expr + odd_coeffs[var] * values.get(var, 0)
                s.add(expr <= 0)
                
                # Apply odd deltas
                for var, delta in odd_deltas.items():
                    values[var] = values.get(var, 0) + delta
        
        if s.check() == sat:
            model = s.model()
            
            even_disjunct = DisjunctInvariant(
                path_condition="i % 2 == 0",
                linear_constraint={
                    var: model.eval(even_coeffs[var]).as_long()
                    for var in var_names
                },
                constant=model.eval(even_coeffs['const']).as_long()
            )
            
            odd_disjunct = DisjunctInvariant(
                path_condition="i % 2 == 1",
                linear_constraint={
                    var: model.eval(odd_coeffs[var]).as_long()
                    for var in var_names
                },
                constant=model.eval(odd_coeffs['const']).as_long()
            )
            
            return DisjunctiveInvariant(
                disjuncts=[even_disjunct, odd_disjunct],
                is_path_sensitive=True
            )
        
        return None
    
    def synthesize_all_forms(self,
                            var_names: List[str],
                            init_values: Dict[str, int],
                            path_analysis: LoopPathAnalysis,
                            max_disjuncts: int = 3) -> List[DisjunctiveInvariant]:
        """
        Try multiple synthesis strategies and return all successful invariants.
        """
        results = []
        
        # Try simple disjunctions
        for n in range(1, max_disjuncts + 1):
            inv = self.synthesize_disjunctive(
                var_names, init_values, path_analysis,
                num_disjuncts=n
            )
            if inv:
                results.append(inv)
        
        # Try path-sensitive if multiple paths
        if len(path_analysis.paths) > 1:
            inv = self.synthesize_path_sensitive(
                var_names, init_values, path_analysis
            )
            if inv:
                results.append(inv)
        
        return results


if __name__ == "__main__":
    from path_analyzer import PathAnalyzer
    
    print("Testing Z3 Disjunctive Solver...")
    print("=" * 60)
    
    solver = Z3DisjunctiveSolver(coeff_bound=10)
    analyzer = PathAnalyzer()
    
    # Test 1: If-then-else with different updates
    print("\nTest 1: If-then-else loop")
    print("  if (i % 2 == 0) { x := x + 2 } else { x := x + 1 }")
    
    analysis = analyzer.analyze_loop(
        "i < n",
        """{ 
            if (i % 2 == 0) {
                x := x + 2;
            } else {
                x := x + 1;
            }
            i := i + 1;
        }""",
        ["i", "x"]
    )
    
    inv = solver.synthesize_disjunctive(
        var_names=["i", "x"],
        init_values={"i": 0, "x": 0},
        path_analysis=analysis,
        num_disjuncts=2
    )
    
    if inv:
        print(f"  Disjunctive: {inv.to_string()}")
    
    # Test 2: Path-sensitive synthesis
    print("\nTest 2: Path-sensitive synthesis")
    
    inv_ps = solver.synthesize_path_sensitive(
        var_names=["i", "x"],
        init_values={"i": 0, "x": 0},
        path_analysis=analysis
    )
    
    if inv_ps:
        print(f"  Path-sensitive: {inv_ps.to_conjunctive_string()}")
    
    # Test 3: Alternating behavior
    print("\nTest 3: Alternating behavior")
    
    inv_alt = solver.synthesize_alternating(
        var_names=["i", "x"],
        init_values={"i": 0, "x": 0},
        even_deltas={"i": 1, "x": 2},
        odd_deltas={"i": 1, "x": 1}
    )
    
    if inv_alt:
        print(f"  Alternating: {inv_alt.to_string()}")
    
    print("\n" + "=" * 60)
    print("Tests completed.")
