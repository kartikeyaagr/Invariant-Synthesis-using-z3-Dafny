"""
Boolean Invariant Synthesis Tool
Synthesizes boolean combinations of linear invariants for Dafny programs.

Supports:
- Single linear invariants: ax + by <= c
- Single linear equalities: ax + by == c
- Conjunctions: (ax + by <= c) && (dx + ey <= f)
- Disjunctions: (ax + by <= c) || (dx + ey <= f)
- Diverse solutions (blocks duplicates)
- Parameter handling (n in requires n >= 0)
"""

import sys
import re
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from dafny_parser import DafnyExtractor
from z3_boolean_solver import (
    Z3BooleanSolver, BooleanInvariant, LinearConstraint, BoolOp, ConstraintType
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
    
    def parse_initial_values(self, method_body: str, var_names: List[str], 
                             parameters: List[str] = None) -> Dict[str, int]:
        """
        Extract initial values from method body before the loop.
        Parameters are treated symbolically (init value 0 for simulation).
        """
        init_values = {}
        parameters = parameters or []
        
        for var in var_names:
            # Parameters don't have initial assignments in body - use 0 for symbolic
            if var in parameters:
                init_values[var] = 0
                continue
            
            # Look for patterns like: var x := 0; or x := 0;
            patterns = [
                rf'var\s+{var}\s*:=\s*(-?\d+)',           # var x := 0
                rf'var\s+{var}\s*:\s*\w+\s*:=\s*(-?\d+)', # var x: int := 0
                rf'\b{var}\s*:=\s*(-?\d+)',               # x := 0
            ]
            
            found = False
            for pattern in patterns:
                match = re.search(pattern, method_body)
                if match:
                    init_values[var] = int(match.group(1))
                    found = True
                    break
            
            if not found:
                init_values[var] = 0
        
        return init_values
    
    def parse_update_expressions(self, loop_body: str, var_names: List[str],
                                 parameters: List[str] = None) -> Dict[str, str]:
        """
        Extract update expressions from loop body.
        Parameters typically don't change inside the loop.
        """
        updates = {}
        parameters = parameters or []
        
        for var in var_names:
            # Parameters usually don't get updated in loop
            if var in parameters:
                updates[var] = "+0"
                continue
            
            # Match: var := var + n, var := var - n, etc.
            patterns = [
                (rf'{var}\s*:=\s*{var}\s*\+\s*(\d+)', '+'),
                (rf'{var}\s*:=\s*{var}\s*-\s*(\d+)', '-'),
                (rf'{var}\s*:=\s*{var}\s*\*\s*(\d+)', '*'),
                (rf'{var}\s*:=\s*(\d+)\s*\+\s*{var}', '+'),
            ]
            
            found = False
            for pattern, op in patterns:
                match = re.search(pattern, loop_body)
                if match:
                    updates[var] = f"{op}{match.group(1)}"
                    found = True
                    break
            
            if not found:
                updates[var] = "+0"
        
        return updates
    
    def extract_method_body(self, source: str, method_name: str) -> str:
        """Extract the full method body from source"""
        pattern = rf'method\s+{method_name}\s*\([^)]*\)'
        match = re.search(pattern, source)
        if not match:
            return ""
        
        rest = source[match.end():]
        brace_pos = rest.find('{')
        if brace_pos == -1:
            return ""
        
        start = match.end() + brace_pos + 1
        brace_count = 1
        end = start
        
        while end < len(source) and brace_count > 0:
            if source[end] == '{':
                brace_count += 1
            elif source[end] == '}':
                brace_count -= 1
            end += 1
        
        return source[start:end-1]
    
    def synthesize_for_loop(self, loop_info: Dict, 
                            method_body: str = "",
                            preconditions: List[str] = None,
                            parameters: List[str] = None) -> List[BooleanInvariant]:
        """Synthesize invariants for a single loop"""
        var_names = loop_info.get('variables', [])
        loop_body = loop_info.get('body', '')
        body_updates = loop_info.get('body_updates', {})
        
        if not var_names:
            return []
        
        # Get initial values (handling parameters specially)
        init_values = self.parse_initial_values(method_body, var_names, parameters)
        
        # Get update expressions - prefer parsed updates from parser
        if body_updates:
            updates = {}
            for var in var_names:
                if var in body_updates:
                    updates[var] = body_updates[var]
                elif var in (parameters or []):
                    updates[var] = "+0"
                else:
                    updates[var] = "+0"
        else:
            updates = self.parse_update_expressions(loop_body, var_names, parameters)
        
        # Use the improved synthesis method
        results = self.solver.synthesize_all_combinations(
            var_names, init_values, updates, 
            max_constraints=self.max_constraints
        )
        
        return results
    
    def synthesize(self, dafny_file: str) -> SynthesisResult:
        """Main synthesis entry point"""
        try:
            with open(dafny_file, 'r') as f:
                source = f.read()
        except Exception as e:
            return SynthesisResult(
                success=False,
                invariants=[],
                validated=[],
                messages=[f"Error reading file: {e}"]
            )
        
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
            
            # Get parameters - handle both tuple format and string format
            raw_params = method.get('parameters', [])
            parameters = []
            for p in raw_params:
                if isinstance(p, tuple):
                    parameters.append(p[0])
                elif isinstance(p, str):
                    parameters.append(p.split(':')[0].strip())
            
            # Also add return variables
            raw_returns = method.get('return_vars', [])
            for r in raw_returns:
                if isinstance(r, tuple):
                    parameters.append(r[0])
                elif isinstance(r, str):
                    parameters.append(r.split(':')[0].strip())
            
            preconditions = method.get('preconditions', [])
            method_body = self.extract_method_body(source, method_name)
            
            for loop_idx, loop in enumerate(method.get('loops', [])):
                messages.append(f"Synthesizing invariants for {method_name}, loop {loop_idx + 1}...")
                
                # Add parameters to loop variables if not already there
                loop_vars = loop.get('variables', [])
                for param in parameters:
                    if param and param not in loop_vars:
                        loop_vars.append(param)
                loop['variables'] = loop_vars
                
                invariants = self.synthesize_for_loop(
                    loop, 
                    method_body,
                    preconditions,
                    parameters
                )
                
                if invariants:
                    messages.append(f"  Found {len(invariants)} candidate invariant(s)")
                    for inv in invariants:
                        inv_str = inv.to_string()
                        all_invariants.append(inv_str)
                        eq_marker = "[EQ]" if inv.constraints[0].is_equality else "[LE]"
                        messages.append(f"    {eq_marker} {inv_str}")
                else:
                    messages.append("  No invariants found")
        
        return SynthesisResult(
            success=len(all_invariants) > 0,
            invariants=all_invariants,
            validated=[False] * len(all_invariants),
            messages=messages
        )
    
    def synthesize_from_spec(self, var_names: List[str],
                             init_values: Dict[str, int],
                             updates: Dict[str, str],
                             require_conjunction: bool = False,
                             require_disjunction: bool = False,
                             require_equality: bool = False) -> List[str]:
        """
        Synthesize invariants from a direct specification.
        """
        results = []
        
        constraint_type = ConstraintType.EQUALITY if require_equality else ConstraintType.INEQUALITY
        
        if require_conjunction:
            for n in range(2, self.max_constraints + 1):
                inv = self.solver.solve_for_coefficients(
                    var_names, init_values, updates,
                    num_constraints=n, 
                    combination_type=BoolOp.AND,
                    constraint_type=constraint_type
                )
                if inv:
                    results.append(inv.to_string())
        elif require_disjunction:
            for n in range(2, self.max_constraints + 1):
                inv = self.solver.solve_for_coefficients(
                    var_names, init_values, updates,
                    num_constraints=n, 
                    combination_type=BoolOp.OR,
                    constraint_type=constraint_type
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
            
            stripped = line.strip()
            if stripped.startswith('while') and '//' not in line.split('while')[0]:
                indent = len(line) - len(line.lstrip())
                inv_indent = ' ' * (indent + 4)
                
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
    parser.add_argument('--equality-only', action='store_true',
                       help='Only synthesize equalities')
    
    args = parser.parse_args()
    
    synthesizer = BooleanInvariantSynthesizer(
        coeff_bound=args.coeff_bound,
        max_constraints=args.max_constraints
    )
    
    result = synthesizer.synthesize(args.input)
    
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
    
    if args.output and result.success:
        inserter = InvariantInserter()
        if inserter.insert_in_file(args.input, args.output, result.invariants):
            print(f"\nWritten to {args.output}")
        else:
            print(f"\nFailed to write to {args.output}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
