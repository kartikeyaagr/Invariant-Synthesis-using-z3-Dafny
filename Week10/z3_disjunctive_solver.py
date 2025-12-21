"""
Z3 Disjunctive Solver for Path-Sensitive Invariant Synthesis
Synthesizes disjunctive invariants of the form:
  (path1_cond && inv1) || (path2_cond && inv2) || ...

UPDATED: Now includes equality synthesis, diverse solutions, and bound invariants.
"""

from z3 import (
    Solver, Int, Bool, Real, And, Or, Not, Implies, If,
    sat, unsat, unknown, simplify, is_true, is_false,
    ForAll, Exists
)
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
from math import gcd
from functools import reduce
import itertools

from path_analyzer import ExecutionPath, LoopPathAnalysis, PathType


class ConstraintType(Enum):
    """Type of constraint to synthesize"""
    INEQUALITY = "inequality"  # <= 0
    EQUALITY = "equality"      # == 0


@dataclass
class DisjunctInvariant:
    """A single disjunct in a disjunctive invariant"""
    path_condition: Optional[str]  # Guard for this disjunct (None = always)
    linear_constraint: Dict[str, int]  # {var: coeff}
    constant: int
    is_equality: bool = False
    
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
            op = "==" if self.is_equality else "<="
            constraint_str = f"0 {op} {-self.constant}"
        else:
            lhs = " + ".join(terms).replace("+ -", "- ")
            op = "==" if self.is_equality else "<="
            constraint_str = f"{lhs} {op} {-self.constant}"
        
        if include_guard and self.path_condition:
            return f"({self.path_condition} ==> {constraint_str})"
        return constraint_str
    
    def is_trivial(self) -> bool:
        """Check if this constraint is trivially true"""
        if all(c == 0 for c in self.linear_constraint.values()):
            if self.is_equality:
                return self.constant == 0
            else:
                return self.constant >= 0
        return False
    
    def signature(self) -> tuple:
        """Return hashable signature for deduplication"""
        coeffs = list(self.linear_constraint.values()) + [self.constant]
        non_zero = [abs(c) for c in coeffs if c != 0]
        
        if not non_zero:
            return (tuple(), 0, self.is_equality, self.path_condition)
        
        g = reduce(gcd, non_zero)
        norm_coeffs = {k: v // g for k, v in self.linear_constraint.items()}
        norm_const = self.constant // g
        
        first_nonzero = None
        for var in sorted(norm_coeffs.keys()):
            if norm_coeffs[var] != 0:
                first_nonzero = norm_coeffs[var]
                break
        
        if first_nonzero and first_nonzero < 0:
            norm_coeffs = {k: -v for k, v in norm_coeffs.items()}
            norm_const = -norm_const
        
        items = tuple(sorted(norm_coeffs.items()))
        return (items, norm_const, self.is_equality, self.path_condition)


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
            return self.disjuncts[0].to_string(include_guard=False)
        
        parts = [d.to_string(include_guard=False) for d in self.disjuncts]
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
    
    def is_trivial(self) -> bool:
        """Check if all disjuncts are trivial"""
        return all(d.is_trivial() for d in self.disjuncts)
    
    def signature(self) -> tuple:
        """Return hashable signature for deduplication"""
        return tuple(d.signature() for d in self.disjuncts)


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
                               loop_bound: int = 20,
                               constraint_type: ConstraintType = ConstraintType.INEQUALITY,
                               blocked_solutions: List[Dict] = None) -> Optional[DisjunctiveInvariant]:
        """
        Synthesize a disjunctive invariant based on path analysis.
        
        Args:
            var_names: Variables in scope
            init_values: Initial values
            path_analysis: Analysis of execution paths
            num_disjuncts: Number of disjuncts to try
            loop_bound: Iterations to simulate
            constraint_type: INEQUALITY (<=) or EQUALITY (==)
            blocked_solutions: Previous solutions to avoid
        
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
        
        # Block previous solutions
        if blocked_solutions:
            for blocked in blocked_solutions:
                block_clause = []
                for d, coeffs in enumerate(coeff_vars):
                    for var in var_names:
                        key = f'a_{d}_{var}'
                        if key in blocked:
                            block_clause.append(coeffs[var] != blocked[key])
                    key = f'c_{d}'
                    if key in blocked:
                        block_clause.append(coeffs['const'] != blocked[key])
                if block_clause:
                    s.add(Or(*block_clause))
        
        # Simulate execution along each path
        for path in path_analysis.paths:
            self._add_path_constraints(
                s, coeff_vars, var_names, init_values, 
                path, loop_bound, constraint_type
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
                    constant=model.eval(coeffs['const']).as_long(),
                    is_equality=(constraint_type == ConstraintType.EQUALITY)
                )
                disjuncts.append(disjunct)
            
            inv = DisjunctiveInvariant(
                disjuncts=disjuncts,
                is_path_sensitive=len(path_analysis.paths) > 1
            )
            
            # Skip trivial - retry with blocking
            if inv.is_trivial():
                new_blocked = list(blocked_solutions) if blocked_solutions else []
                solution_dict = {}
                for d, coeffs in enumerate(coeff_vars):
                    for var in var_names:
                        solution_dict[f'a_{d}_{var}'] = model.eval(coeffs[var]).as_long()
                    solution_dict[f'c_{d}'] = model.eval(coeffs['const']).as_long()
                new_blocked.append(solution_dict)
                
                if len(new_blocked) < 20:
                    return self.synthesize_disjunctive(
                        var_names, init_values, path_analysis,
                        num_disjuncts, loop_bound, constraint_type, new_blocked
                    )
                return None
            
            return inv
        
        return None
    
    def _add_path_constraints(self, solver: Solver,
                              coeff_vars: List[Dict],
                              var_names: List[str],
                              init_values: Dict[str, int],
                              path: ExecutionPath,
                              loop_bound: int,
                              constraint_type: ConstraintType = ConstraintType.INEQUALITY):
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
                
                if constraint_type == ConstraintType.EQUALITY:
                    disjunct_constraints.append(expr == 0)
                else:
                    disjunct_constraints.append(expr <= 0)
            
            # Add: at least one disjunct satisfied
            solver.add(Or(*disjunct_constraints))
    
    def synthesize_path_sensitive(self,
                                  var_names: List[str],
                                  init_values: Dict[str, int],
                                  path_analysis: LoopPathAnalysis,
                                  loop_bound: int = 20,
                                  constraint_type: ConstraintType = ConstraintType.INEQUALITY) -> Optional[DisjunctiveInvariant]:
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
                
                if constraint_type == ConstraintType.EQUALITY:
                    s.add(expr == 0)
                else:
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
                    constant=model.eval(coeffs['const']).as_long(),
                    is_equality=(constraint_type == ConstraintType.EQUALITY)
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
    
    def synthesize_simple(self,
                          var_names: List[str],
                          init_values: Dict[str, int],
                          update_exprs: Dict[str, str],
                          loop_bound: int = 20,
                          constraint_type: ConstraintType = ConstraintType.INEQUALITY,
                          blocked_solutions: List[Dict] = None) -> Optional[DisjunctInvariant]:
        """
        Synthesize a single simple invariant (not disjunctive).
        Used for linear loops without conditionals.
        """
        coeff_vars = {var: Int(f'a_{var}') for var in var_names}
        coeff_vars['const'] = Int('c')
        
        s = Solver()
        
        # Bounds
        for v in coeff_vars.values():
            s.add(v >= -self.coeff_bound, v <= self.coeff_bound)
        
        # Non-triviality
        var_coeffs = [coeff_vars[v] for v in var_names]
        s.add(Or(*[c != 0 for c in var_coeffs]))
        
        # Block previous solutions
        if blocked_solutions:
            for blocked in blocked_solutions:
                block_clause = []
                for var in var_names:
                    if var in blocked:
                        block_clause.append(coeff_vars[var] != blocked[var])
                if 'const' in blocked:
                    block_clause.append(coeff_vars['const'] != blocked['const'])
                if block_clause:
                    s.add(Or(*block_clause))
        
        # Simulate
        for step in range(loop_bound + 1):
            values = {}
            for var in var_names:
                base_val = init_values.get(var, 0)
                update = update_exprs.get(var, "+0")
                
                if step == 0:
                    values[var] = base_val
                else:
                    if update.startswith('+'):
                        try:
                            delta = int(update[1:])
                            values[var] = base_val + step * delta
                        except:
                            values[var] = base_val
                    elif update.startswith('-'):
                        try:
                            delta = int(update[1:])
                            values[var] = base_val - step * delta
                        except:
                            values[var] = base_val
                    else:
                        values[var] = base_val
            
            expr = coeff_vars['const']
            for var in var_names:
                expr = expr + coeff_vars[var] * values[var]
            
            if constraint_type == ConstraintType.EQUALITY:
                s.add(expr == 0)
            else:
                s.add(expr <= 0)
        
        if s.check() == sat:
            model = s.model()
            inv = DisjunctInvariant(
                path_condition=None,
                linear_constraint={var: model.eval(coeff_vars[var]).as_long() for var in var_names},
                constant=model.eval(coeff_vars['const']).as_long(),
                is_equality=(constraint_type == ConstraintType.EQUALITY)
            )
            
            if inv.is_trivial():
                new_blocked = list(blocked_solutions) if blocked_solutions else []
                solution_dict = {var: model.eval(coeff_vars[var]).as_long() for var in var_names}
                solution_dict['const'] = model.eval(coeff_vars['const']).as_long()
                new_blocked.append(solution_dict)
                
                if len(new_blocked) < 20:
                    return self.synthesize_simple(
                        var_names, init_values, update_exprs,
                        loop_bound, constraint_type, new_blocked
                    )
                return None
            
            return inv
        
        return None
    
    def synthesize_diverse_simple(self,
                                  var_names: List[str],
                                  init_values: Dict[str, int],
                                  update_exprs: Dict[str, str],
                                  num_solutions: int = 5,
                                  constraint_type: ConstraintType = ConstraintType.INEQUALITY,
                                  loop_bound: int = 20) -> List[DisjunctInvariant]:
        """Synthesize multiple diverse simple invariants."""
        results = []
        blocked = []
        seen_signatures = set()
        
        attempts = 0
        max_attempts = num_solutions * 3
        
        while len(results) < num_solutions and attempts < max_attempts:
            attempts += 1
            
            inv = self.synthesize_simple(
                var_names, init_values, update_exprs,
                loop_bound, constraint_type, blocked
            )
            
            if inv is None:
                break
            
            sig = inv.signature()
            if sig not in seen_signatures and not inv.is_trivial():
                seen_signatures.add(sig)
                results.append(inv)
            
            # Block this solution
            solution_dict = {var: inv.linear_constraint[var] for var in var_names}
            solution_dict['const'] = inv.constant
            blocked.append(solution_dict)
        
        return results
    
    def synthesize_bound_invariants(self, var_names: List[str],
                                    init_values: Dict[str, int],
                                    update_exprs: Dict[str, str]) -> List[DisjunctInvariant]:
        """Synthesize bound invariants like i >= 0, i <= n"""
        results = []
        
        for var in var_names:
            update = update_exprs.get(var, "+0")
            init_val = init_values.get(var, 0)
            
            # Skip likely parameters
            if update == "+0" and var not in ['i', 'j', 'x', 'y', 'z', 'k', 'count', 'sum', 'result']:
                continue
            
            # x >= 0 if starts at 0 and increases
            if init_val == 0 and update.startswith('+'):
                inv = DisjunctInvariant(
                    path_condition=None,
                    linear_constraint={v: -1 if v == var else 0 for v in var_names},
                    constant=0,
                    is_equality=False
                )
                results.append(inv)
            
            # x <= 0 if starts at 0 and decreases
            if init_val == 0 and update.startswith('-'):
                inv = DisjunctInvariant(
                    path_condition=None,
                    linear_constraint={v: 1 if v == var else 0 for v in var_names},
                    constant=0,
                    is_equality=False
                )
                results.append(inv)
        
        # Try upper bounds like i <= n
        for var in var_names:
            update = update_exprs.get(var, "+0")
            if update == "+0":
                continue
                
            for param in var_names:
                if update_exprs.get(param, "+0") != "+0":
                    continue
                
                if init_values.get(var, 0) == 0 and update.startswith('+'):
                    inv = DisjunctInvariant(
                        path_condition=None,
                        linear_constraint={v: (1 if v == var else (-1 if v == param else 0)) for v in var_names},
                        constant=0,
                        is_equality=False
                    )
                    results.append(inv)
        
        return results

    def synthesize_all_forms(self,
                            var_names: List[str],
                            init_values: Dict[str, int],
                            path_analysis: LoopPathAnalysis,
                            max_disjuncts: int = 3,
                            update_exprs: Dict[str, str] = None) -> List[DisjunctiveInvariant]:
        """
        Try multiple synthesis strategies and return all successful invariants.
        Now includes equalities, diverse solutions, and bound invariants.
        """
        results = []
        seen_signatures = set()
        
        def add_if_unique(inv):
            if inv and not inv.is_trivial():
                sig = inv.signature()
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    results.append(inv)
        
        # Build update_exprs from path analysis if not provided
        if update_exprs is None:
            update_exprs = {}
            for path in path_analysis.paths:
                for var, upd in path.updates.items():
                    if upd.delta is not None and var not in update_exprs:
                        update_exprs[var] = f"+{upd.delta}" if upd.delta >= 0 else str(upd.delta)
            for var in var_names:
                if var not in update_exprs:
                    update_exprs[var] = "+0"
        
        # 1. Diverse simple equalities
        diverse_eq = self.synthesize_diverse_simple(
            var_names, init_values, update_exprs,
            num_solutions=5,
            constraint_type=ConstraintType.EQUALITY
        )
        for inv in diverse_eq:
            add_if_unique(DisjunctiveInvariant(disjuncts=[inv], is_path_sensitive=False))
        
        # 2. Diverse simple inequalities
        diverse_ineq = self.synthesize_diverse_simple(
            var_names, init_values, update_exprs,
            num_solutions=5,
            constraint_type=ConstraintType.INEQUALITY
        )
        for inv in diverse_ineq:
            add_if_unique(DisjunctiveInvariant(disjuncts=[inv], is_path_sensitive=False))
        
        # 3. Bound invariants
        bound_invs = self.synthesize_bound_invariants(var_names, init_values, update_exprs)
        for inv in bound_invs:
            add_if_unique(DisjunctiveInvariant(disjuncts=[inv], is_path_sensitive=False))
        
        # 4. Disjunctive invariants (inequalities)
        for n in range(2, max_disjuncts + 1):
            inv = self.synthesize_disjunctive(
                var_names, init_values, path_analysis,
                num_disjuncts=n,
                constraint_type=ConstraintType.INEQUALITY
            )
            add_if_unique(inv)
        
        # 5. Disjunctive invariants (equalities)
        for n in range(2, max_disjuncts + 1):
            inv = self.synthesize_disjunctive(
                var_names, init_values, path_analysis,
                num_disjuncts=n,
                constraint_type=ConstraintType.EQUALITY
            )
            add_if_unique(inv)
        
        # 6. Path-sensitive if multiple paths
        if len(path_analysis.paths) > 1:
            inv = self.synthesize_path_sensitive(
                var_names, init_values, path_analysis,
                constraint_type=ConstraintType.INEQUALITY
            )
            add_if_unique(inv)
            
            inv = self.synthesize_path_sensitive(
                var_names, init_values, path_analysis,
                constraint_type=ConstraintType.EQUALITY
            )
            add_if_unique(inv)
        
        return results


if __name__ == "__main__":
    from path_analyzer import PathAnalyzer
    
    print("Testing Z3 Disjunctive Solver...")
    print("=" * 60)
    
    solver = Z3DisjunctiveSolver(coeff_bound=10)
    analyzer = PathAnalyzer()
    
    # Test 1: Simple linear loop
    print("\nTest 1: Simple linear loop (j == 2*i)")
    print("  i := 0; j := 0; while i < n: i++; j += 2")
    
    analysis = analyzer.analyze_loop("i < n", "{ i := i + 1; j := j + 2; }", ["i", "j", "n"])
    
    all_invs = solver.synthesize_all_forms(
        var_names=["i", "j", "n"],
        init_values={"i": 0, "j": 0, "n": 0},
        path_analysis=analysis
    )
    
    print(f"  Found {len(all_invs)} invariants:")
    for inv in all_invs[:10]:
        eq_marker = "[EQ]" if inv.disjuncts[0].is_equality else "[LE]"
        print(f"    {eq_marker} {inv.to_string()}")
    
    # Test 2: If-then-else with different updates
    print("\nTest 2: If-then-else loop")
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
    
    # Test 3: Path-sensitive synthesis
    print("\nTest 3: Path-sensitive synthesis")
    
    inv_ps = solver.synthesize_path_sensitive(
        var_names=["i", "x"],
        init_values={"i": 0, "x": 0},
        path_analysis=analysis
    )
    
    if inv_ps:
        print(f"  Path-sensitive: {inv_ps.to_conjunctive_string()}")
    
    # Test 4: All forms for if-else
    print("\nTest 4: All forms for if-else loop")
    all_invs = solver.synthesize_all_forms(
        var_names=["i", "x", "n"],
        init_values={"i": 0, "x": 0, "n": 0},
        path_analysis=analysis
    )
    
    print(f"  Found {len(all_invs)} invariants:")
    for inv in all_invs[:8]:
        if inv.is_path_sensitive:
            print(f"    [PS] {inv.to_conjunctive_string()}")
        else:
            eq_marker = "[EQ]" if inv.disjuncts[0].is_equality else "[LE]"
            print(f"    {eq_marker} {inv.to_string()}")
    
    print("\n" + "=" * 60)
    print("Tests completed.")
