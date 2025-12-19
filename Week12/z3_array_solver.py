"""
Z3 Solver for Array Invariants
Synthesizes invariants involving arrays and quantified properties.
"""

from z3 import (
    Solver, Int, Bool, Array, IntSort, And, Or, Not, Implies, ForAll, Exists,
    sat, unsat, Select, Store, If, simplify
)
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum

from array_analyzer import (
    ArrayAnalyzer, LoopArrayAnalysis, ArrayPattern, 
    ArrayInvariantTemplates, QuantifiedInvariant
)


@dataclass
class ArrayInvariant:
    """Represents a synthesized array invariant"""
    invariant_type: str  # "bounds", "quantified", "accumulator", "relationship"
    expression: str      # Dafny-compatible string
    confidence: float    # 0.0 to 1.0
    
    def to_string(self) -> str:
        return self.expression


class Z3ArraySolver:
    """
    Z3-based solver for array invariant synthesis.
    
    Handles:
    1. Array bounds invariants
    2. Quantified invariants (forall, exists)
    3. Accumulator invariants
    4. Element relationships
    """
    
    def __init__(self, coeff_bound: int = 10):
        self.coeff_bound = coeff_bound
        self.templates = ArrayInvariantTemplates()
    
    def synthesize_bounds_invariants(self, analysis: LoopArrayAnalysis) -> List[ArrayInvariant]:
        """Synthesize bounds invariants for array indices"""
        invariants = []
        
        idx = analysis.index_var
        bound = analysis.index_bound
        
        # Basic index bounds
        if analysis.index_direction == "increasing":
            invariants.append(ArrayInvariant(
                invariant_type="bounds",
                expression=f"0 <= {idx}",
                confidence=1.0
            ))
            invariants.append(ArrayInvariant(
                invariant_type="bounds",
                expression=f"{idx} <= {bound}",
                confidence=1.0
            ))
        else:
            invariants.append(ArrayInvariant(
                invariant_type="bounds",
                expression=f"{idx} >= 0",
                confidence=1.0
            ))
            invariants.append(ArrayInvariant(
                invariant_type="bounds",
                expression=f"{idx} < {bound}",
                confidence=0.9
            ))
        
        # Array-specific bounds
        for arr_name in analysis.arrays:
            invariants.append(ArrayInvariant(
                invariant_type="bounds",
                expression=f"0 <= {idx} <= {arr_name}.Length",
                confidence=0.95
            ))
        
        return invariants
    
    def synthesize_quantified_invariants(self, analysis: LoopArrayAnalysis,
                                         element_property: str = None) -> List[ArrayInvariant]:
        """Synthesize quantified invariants based on array pattern"""
        invariants = []
        
        idx = analysis.index_var
        
        # Based on pattern, generate appropriate quantified invariants
        if analysis.pattern == ArrayPattern.INIT:
            # Initialization pattern
            for arr in analysis.modified_arrays:
                # Try to detect initialization value
                for access in analysis.accesses:
                    if access.array == arr and access.value:
                        inv = self.templates.initialized_elements(idx, arr, access.value)
                        invariants.append(ArrayInvariant(
                            invariant_type="quantified",
                            expression=inv.to_string(),
                            confidence=0.9
                        ))
        
        elif analysis.pattern == ArrayPattern.COPY:
            # Copy pattern
            src_arrays = analysis.read_arrays - analysis.modified_arrays
            dst_arrays = analysis.modified_arrays
            
            for src in src_arrays:
                for dst in dst_arrays:
                    inv = self.templates.copy_invariant(src, dst, idx)
                    invariants.append(ArrayInvariant(
                        invariant_type="quantified",
                        expression=inv.to_string(),
                        confidence=0.85
                    ))
        
        elif analysis.pattern == ArrayPattern.TRANSFORM:
            # Transform pattern - elements processed somehow
            for arr in analysis.modified_arrays:
                # Generic "processed elements" invariant
                inv = QuantifiedInvariant(
                    quantifier="forall",
                    bound_var="k",
                    lower_bound="0",
                    upper_bound=idx,
                    property=f"{arr}[k] >= 0"  # Default property
                )
                invariants.append(ArrayInvariant(
                    invariant_type="quantified",
                    expression=inv.to_string(),
                    confidence=0.5
                ))
        
        elif analysis.pattern == ArrayPattern.LINEAR_SCAN:
            # Linear scan - processed elements examined
            for arr in analysis.read_arrays:
                # Add generic "examined up to i" style invariant
                pass  # Often needs specific property
        
        # Add user-specified property if provided
        if element_property:
            for arr in analysis.arrays:
                inv = QuantifiedInvariant(
                    quantifier="forall",
                    bound_var="k",
                    lower_bound="0",
                    upper_bound=idx,
                    property=element_property.replace("$arr", arr).replace("$k", "k")
                )
                invariants.append(ArrayInvariant(
                    invariant_type="quantified",
                    expression=inv.to_string(),
                    confidence=0.7
                ))
        
        return invariants
    
    def synthesize_accumulator_invariants(self, analysis: LoopArrayAnalysis,
                                          acc_var: str = None) -> List[ArrayInvariant]:
        """Synthesize accumulator-style invariants"""
        invariants = []
        
        if analysis.pattern != ArrayPattern.ACCUMULATE:
            return invariants
        
        idx = analysis.index_var
        
        # Look for accumulator variable
        if acc_var is None:
            # Common accumulator names
            for candidate in ["sum", "total", "count", "product", "acc", "result"]:
                # Would need to verify from code
                acc_var = candidate
                break
        
        if acc_var:
            for arr in analysis.read_arrays:
                # Sum invariant
                invariants.append(ArrayInvariant(
                    invariant_type="accumulator",
                    expression=f"{acc_var} >= 0",
                    confidence=0.8
                ))
                
                # More specific: sum equals Sum of slice
                invariants.append(ArrayInvariant(
                    invariant_type="accumulator",
                    expression=f"{acc_var} == SumUpto({arr}, {idx})",
                    confidence=0.6
                ))
        
        return invariants
    
    def synthesize_relationship_invariants(self, analysis: LoopArrayAnalysis,
                                           variables: List[str]) -> List[ArrayInvariant]:
        """Synthesize linear relationships between variables and array properties"""
        invariants = []
        
        idx = analysis.index_var
        
        # Try linear relationships between index and other variables
        for var in variables:
            if var != idx and var not in analysis.arrays:
                # Try: var == k * idx for small k
                for k in [1, 2, -1]:
                    if k == 1:
                        invariants.append(ArrayInvariant(
                            invariant_type="relationship",
                            expression=f"{var} == {idx}",
                            confidence=0.3
                        ))
                    elif k == -1:
                        invariants.append(ArrayInvariant(
                            invariant_type="relationship",
                            expression=f"{var} + {idx} == 0",
                            confidence=0.3
                        ))
                    else:
                        invariants.append(ArrayInvariant(
                            invariant_type="relationship",
                            expression=f"{var} == {k} * {idx}",
                            confidence=0.3
                        ))
        
        return invariants
    
    def synthesize_all(self, analysis: LoopArrayAnalysis,
                       variables: List[str] = None,
                       element_property: str = None,
                       acc_var: str = None) -> List[ArrayInvariant]:
        """Synthesize all types of invariants"""
        all_invariants = []
        
        # Bounds
        all_invariants.extend(self.synthesize_bounds_invariants(analysis))
        
        # Quantified
        all_invariants.extend(self.synthesize_quantified_invariants(
            analysis, element_property
        ))
        
        # Accumulator
        all_invariants.extend(self.synthesize_accumulator_invariants(
            analysis, acc_var
        ))
        
        # Relationships
        if variables:
            all_invariants.extend(self.synthesize_relationship_invariants(
                analysis, variables
            ))
        
        # Sort by confidence
        all_invariants.sort(key=lambda x: x.confidence, reverse=True)
        
        return all_invariants
    
    def verify_with_z3(self, invariant: str, 
                       index_var: str, array_var: str,
                       init_index: int, bound: int) -> bool:
        """
        Attempt to verify an invariant using Z3.
        This is a simplified verification for non-quantified invariants.
        """
        s = Solver()
        
        # Create Z3 variables
        idx = Int(index_var)
        arr = Array(array_var, IntSort(), IntSort())
        
        # Parse and verify (simplified)
        # This would need a proper parser for full functionality
        
        return True  # Placeholder


class QuantifiedInvariantSynthesizer:
    """
    Specialized synthesizer for quantified array invariants.
    Uses pattern matching and template instantiation.
    """
    
    def __init__(self):
        self.templates = ArrayInvariantTemplates()
    
    def synthesize_forall(self, array: str, index: str,
                          element_expr: str) -> List[str]:
        """
        Synthesize forall invariants.
        
        Args:
            array: Array variable name
            index: Current loop index
            element_expr: Expression template with $k for bound var
        """
        results = []
        
        # Standard forall over processed elements
        prop = element_expr.replace("$k", "k")
        results.append(
            f"forall k :: 0 <= k < {index} ==> {prop}"
        )
        
        # With array access
        if "$arr" in element_expr:
            prop = element_expr.replace("$k", "k").replace("$arr", f"{array}[k]")
            results.append(
                f"forall k :: 0 <= k < {index} ==> {prop}"
            )
        
        return results
    
    def synthesize_exists(self, array: str, index: str,
                          element_expr: str) -> List[str]:
        """Synthesize exists invariants"""
        results = []
        
        prop = element_expr.replace("$k", "k")
        results.append(
            f"exists k :: 0 <= k < {index} && {prop}"
        )
        
        return results
    
    def synthesize_sorted(self, array: str, index: str) -> List[str]:
        """Synthesize sortedness invariants"""
        return [
            f"forall j, k :: 0 <= j < k < {index} ==> {array}[j] <= {array}[k]"
        ]
    
    def synthesize_partition(self, array: str, pivot_idx: str,
                            pivot_val: str) -> List[str]:
        """Synthesize partition invariants"""
        return [
            f"forall k :: 0 <= k < {pivot_idx} ==> {array}[k] <= {pivot_val}",
            f"forall k :: {pivot_idx} < k < {array}.Length ==> {array}[k] > {pivot_val}"
        ]
    
    def synthesize_permutation(self, array: str, old_array: str) -> List[str]:
        """Synthesize multiset equality (permutation)"""
        return [
            f"multiset({array}[..]) == multiset({old_array}[..])"
        ]


if __name__ == "__main__":
    print("Testing Z3 Array Solver...")
    print("=" * 60)
    
    from array_analyzer import ArrayAnalyzer
    
    analyzer = ArrayAnalyzer()
    solver = Z3ArraySolver()
    
    # Test 1: Array initialization
    print("\nTest 1: Array initialization invariants")
    analysis = analyzer.analyze_loop(
        "i < a.Length",
        "{ a[i] := 0; i := i + 1; }",
        ["i", "a"],
        ["a"]
    )
    
    invariants = solver.synthesize_all(analysis, ["i", "a"])
    print(f"  Found {len(invariants)} invariants:")
    for inv in invariants[:5]:
        print(f"    [{inv.confidence:.1f}] {inv.expression}")
    
    # Test 2: Array copy
    print("\nTest 2: Array copy invariants")
    analysis = analyzer.analyze_loop(
        "i < src.Length",
        "{ dst[i] := src[i]; i := i + 1; }",
        ["i", "src", "dst"],
        ["src", "dst"]
    )
    
    invariants = solver.synthesize_all(analysis, ["i", "src", "dst"])
    print(f"  Found {len(invariants)} invariants:")
    for inv in invariants[:5]:
        print(f"    [{inv.confidence:.1f}] {inv.expression}")
    
    # Test 3: Quantified synthesizer
    print("\nTest 3: Quantified invariant templates")
    quant_synth = QuantifiedInvariantSynthesizer()
    
    forall_invs = quant_synth.synthesize_forall("a", "i", "$arr >= 0")
    print(f"  Forall invariants:")
    for inv in forall_invs:
        print(f"    {inv}")
    
    sorted_invs = quant_synth.synthesize_sorted("a", "i")
    print(f"  Sorted invariants:")
    for inv in sorted_invs:
        print(f"    {inv}")
    
    print("\n" + "=" * 60)
    print("Tests completed.")
