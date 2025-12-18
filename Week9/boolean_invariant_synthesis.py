"""
Boolean Invariant Synthesis Tool
Synthesizes boolean combinations of linear invariants for Dafny programs.

Supports:
- Single linear invariants: ax + by <= c
- Conjunctions: (ax + by <= c) && (dx + ey <= f)
- Disjunctions: (ax + by <= c) || (dx + ey <= f)
- Mixed: (ax + by <= c) && ((dx + ey <= f) || (gx + hy <= i))
"""

import sys
import re
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from dafny_parser import DafnyExtractor
from z3_boolean_solver import (
    Z3BooleanSolver, BooleanInvariant, LinearConstraint, BoolOp
)


@dataclass
class SynthesisResult:
    """Result of invariant synthesis"""
    success: bool
    invariants: List[str]
    validated: List[bool]
    messages: List[str]


class BooleanInvariantSynthesizer:
    """
    Main synthesizer class for boolean combinations of linear invariants.
    """
    
    def __init__(self, coeff_bound: int = 10, max_constraints: int = 3):
        self.parser = DafnyExtractor()
        self.solver = Z3BooleanSolver(coeff_bound=coeff_bound)
        self.max_constraints = max_constraints
    
    def parse_initial_values(self, method_body: str, var_names: List[str]) -> Dict[str, int]:
        """Extract initial values from method body before the loop"""
        init_values = {}
        
        for var in var_names:
            # Look for patterns like: var := 0; or x := 0;
            patterns = [
                rf'{var}\s*:=\s*(-?\d+)',
                rf'var\s+{var}\s*:=\s*(-?\d+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, method_body)
                if match:
                    init_values[var] = int(match.group(1))
                    break
            
            if var not in init_values:
                init_values[var] = 0
        
        return init_values
    
    def parse_update_expressions(self, loop_body: str, var_names: List[str]) -> Dict[str, str]:
        """Extract update expressions from loop body"""
        updates = {}
        
        for var in var_names:
            # Match: var := var + n, var := var - n, etc.
            patterns = [
                (rf'{var}\s*:=\s*{var}\s*\+\s*(\d+)', '+'),
                (rf'{var}\s*:=\s*{var}\s*-\s*(\d+)', '-'),
                (rf'{var}\s*:=\s*{var}\s*\*\s*(\d+)', '*'),
                (rf'{var}\s*:=\s*(\d+)\s*\+\s*{var}', '+'),
            ]
            
            for pattern, op in patterns:
                match = re.search(pattern, loop_body)
                if match:
                    updates[var] = f"{op}{match.group(1)}"
                    break
            
            if var not in updates:
                updates[var] = "+0"
        
        return updates
    
    def synthesize_for_loop(self, loop_info: Dict, 
                            method_body: str = "",
                            preconditions: List[str] = None) -> List[BooleanInvariant]:
        """Synthesize invariants for a single loop"""
        var_names = loop_info.get('variables', [])
        loop_body = loop_info.get('body', '')
        body_updates = loop_info.get('body_updates', {})
        
        if not var_names:
            return []
        
        # Get initial values
        init_values = self.parse_initial_values(method_body, var_names)
        
        # Get update expressions
        if body_updates:
            updates = body_updates
        else:
            updates = self.parse_update_expressions(loop_body, var_names)
        
        # Synthesize invariants
        results = []
        
        # Try single constraints
        inv = self.solver.solve_for_coefficients(
            var_names, init_values, updates,
            num_constraints=1
        )
        if inv:
            results.append(inv)
        
        # Try conjunctions of increasing size
        for n in range(2, self.max_constraints + 1):
            inv_conj = self.solver.solve_for_coefficients(
                var_names, init_values, updates,
                num_constraints=n, combination_type=BoolOp.AND
            )
            if inv_conj:
                results.append(inv_conj)
        
        # Try disjunctions of increasing size
        for n in range(2, self.max_constraints + 1):
            inv_disj = self.solver.solve_for_coefficients(
                var_names, init_values, updates,
                num_constraints=n, combination_type=BoolOp.OR
            )
            if inv_disj:
                results.append(inv_disj)
        
        return results
    
    def synthesize(self, dafny_file: str) -> SynthesisResult:
        """Main synthesis entry point"""
        # Parse the Dafny file
        parsed = self.parser.parse_file(dafny_file)
        
        if "error" in parsed:
            return SynthesisResult(
                success=False,
                invariants=[],
                validated=[],
                messages=[f"Parse error: {parsed['error']}"]
            )
        
        all_invariants = []
        messages = []
        
        for method in parsed.get('methods', []):
            method_name = method.get('name', 'unknown')
            
            for loop_idx, loop in enumerate(method.get('loops', [])):
                messages.append(f"Synthesizing invariants for {method_name}, loop {loop_idx + 1}...")
                
                # Get method body for initial value extraction
                # This is a simplification - in practice, we'd need the full body
                method_body = ""
                
                invariants = self.synthesize_for_loop(
                    loop, 
                    method_body,
                    method.get('preconditions', [])
                )
                
                if invariants:
                    messages.append(f"  Found {len(invariants)} candidate invariant(s)")
                    for inv in invariants:
                        inv_str = inv.to_string()
                        all_invariants.append(inv_str)
                        messages.append(f"    - {inv_str}")
                else:
                    messages.append("  No invariants found")
        
        return SynthesisResult(
            success=len(all_invariants) > 0,
            invariants=all_invariants,
            validated=[False] * len(all_invariants),  # Validation done separately
            messages=messages
        )
    
    def synthesize_from_spec(self, var_names: List[str],
                             init_values: Dict[str, int],
                             updates: Dict[str, str],
                             require_conjunction: bool = False,
                             require_disjunction: bool = False) -> List[str]:
        """
        Synthesize invariants from a direct specification.
        Useful for testing and direct API access.
        """
        results = []
        
        if require_conjunction:
            for n in range(2, self.max_constraints + 1):
                inv = self.solver.solve_for_coefficients(
                    var_names, init_values, updates,
                    num_constraints=n, combination_type=BoolOp.AND
                )
                if inv:
                    results.append(inv.to_string())
        elif require_disjunction:
            for n in range(2, self.max_constraints + 1):
                inv = self.solver.solve_for_coefficients(
                    var_names, init_values, updates,
                    num_constraints=n, combination_type=BoolOp.OR
                )
                if inv:
                    results.append(inv.to_string())
        else:
            all_invs = self.solver.synthesize_all_combinations(
                var_names, init_values, updates, self.max_constraints
            )
            results = [inv.to_string() for inv in all_invs]
        
        return results


class InvariantInserter:
    """Inserts synthesized invariants into Dafny programs"""
    
    def insert(self, source: str, invariants: List[str]) -> str:
        """Insert invariants after while loop declaration"""
        lines = source.split('\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            
            # Check if this is a while line
            if 'while' in line and '{' not in line:
                # Find indentation
                indent = len(line) - len(line.lstrip())
                inv_indent = ' ' * (indent + 2)
                
                # Insert invariants
                for inv in invariants:
                    new_lines.append(f"{inv_indent}invariant {inv}")
        
        return '\n'.join(new_lines)
    
    def insert_in_file(self, input_file: str, output_file: str, 
                       invariants: List[str]) -> bool:
        """Insert invariants and write to new file"""
        try:
            with open(input_file, 'r') as f:
                source = f.read()
            
            modified = self.insert(source, invariants)
            
            with open(output_file, 'w') as f:
                f.write(modified)
            
            return True
        except Exception as e:
            print(f"Error inserting invariants: {e}")
            return False


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Boolean Invariant Synthesis Tool'
    )
    parser.add_argument('input', help='Input Dafny file')
    parser.add_argument('-o', '--output', help='Output file with invariants')
    parser.add_argument('-c', '--coeff-bound', type=int, default=10,
                       help='Coefficient bound for synthesis (default: 10)')
    parser.add_argument('-n', '--max-constraints', type=int, default=3,
                       help='Maximum constraints to combine (default: 3)')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    parser.add_argument('--conjunction-only', action='store_true',
                       help='Only synthesize conjunctions')
    parser.add_argument('--disjunction-only', action='store_true',
                       help='Only synthesize disjunctions')
    
    args = parser.parse_args()
    
    # Create synthesizer
    synthesizer = BooleanInvariantSynthesizer(
        coeff_bound=args.coeff_bound,
        max_constraints=args.max_constraints
    )
    
    # Run synthesis
    result = synthesizer.synthesize(args.input)
    
    # Output results
    if args.json:
        output = {
            "success": result.success,
            "invariants": result.invariants,
            "messages": result.messages
        }
        print(json.dumps(output, indent=2))
    else:
        for msg in result.messages:
            print(msg)
        
        if result.success:
            print(f"\nSynthesized {len(result.invariants)} invariant(s)")
        else:
            print("\nNo invariants found")
    
    # Write output file if requested
    if args.output and result.success:
        inserter = InvariantInserter()
        if inserter.insert_in_file(args.input, args.output, result.invariants):
            print(f"\nWritten to {args.output}")
        else:
            print(f"\nFailed to write to {args.output}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
