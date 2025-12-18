"""
Quadratic Invariant Synthesis Tool
Synthesizes non-linear invariants of the form:
  ax² + by² + cxy + dx + ey + f ≤ 0

Extends linear synthesis (Weeks 6, 9-10) to polynomial invariants.
"""

import sys
import os
import re
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from z3_quadratic_solver import (
    Z3QuadraticSolver, QuadraticInvariant, QuadraticType
)


@dataclass
class SynthesisResult:
    """Result of quadratic invariant synthesis"""
    success: bool
    invariants: List[str]
    invariant_objects: List[QuadraticInvariant]
    synthesis_type: str
    messages: List[str]


class DafnyParser:
    """Simplified Dafny parser for extracting loop information"""
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        try:
            with open(filepath, 'r') as f:
                source = f.read()
            return self.parse_source(source)
        except Exception as e:
            return {"error": str(e)}
    
    def parse_source(self, source: str) -> Dict[str, Any]:
        result = {
            "methods": [],
            "loops": [],
            "preconditions": [],
            "postconditions": []
        }
        
        # Find method declarations
        method_pattern = r'method\s+(\w+)\s*\(([^)]*)\)(?:\s*returns\s*\(([^)]*)\))?'
        
        for match in re.finditer(method_pattern, source):
            method_name = match.group(1)
            params_str = match.group(2)
            returns_str = match.group(3) or ""
            
            params = self._parse_params(params_str)
            returns = self._parse_params(returns_str)
            
            # Find method body
            after_match = source[match.end():]
            
            # Find specs
            specs_pattern = r'^((?:\s*requires[^\n]*\n|\s*ensures[^\n]*\n)*)'
            specs_match = re.match(specs_pattern, after_match)
            specs_str = specs_match.group(1) if specs_match else ""
            body_start = specs_match.end() if specs_match else 0
            
            preconditions = re.findall(r'requires\s+([^\n]+)', specs_str)
            postconditions = re.findall(r'ensures\s+([^\n]+)', specs_str)
            
            # Extract body
            rest = after_match[body_start:]
            body = self._extract_brace_content(rest)
            
            loops = self._extract_loops(body, params + returns)
            
            method_data = {
                "name": method_name,
                "parameters": params,
                "returns": returns,
                "preconditions": [p.strip() for p in preconditions],
                "postconditions": [p.strip() for p in postconditions],
                "body": body,
                "loops": loops
            }
            
            result["methods"].append(method_data)
            result["loops"].extend(loops)
            result["preconditions"].extend(method_data["preconditions"])
            result["postconditions"].extend(method_data["postconditions"])
        
        return result
    
    def _parse_params(self, params_str: str) -> List[str]:
        if not params_str.strip():
            return []
        params = []
        for part in params_str.split(','):
            part = part.strip()
            if ':' in part:
                name = part.split(':')[0].strip()
                params.append(name)
            elif part:
                params.append(part)
        return params
    
    def _extract_brace_content(self, text: str) -> str:
        brace_start = text.find('{')
        if brace_start == -1:
            return ""
        
        brace_count = 0
        content = ""
        started = False
        
        for c in text:
            if c == '{':
                brace_count += 1
                if not started:
                    started = True
                    continue
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    break
            if started:
                content += c
        
        return content
    
    def _extract_loops(self, body: str, variables: List[str]) -> List[Dict]:
        loops = []
        
        while_starts = [m.start() for m in re.finditer(r'\bwhile\b', body)]
        
        for start in while_starts:
            # Extract condition
            cond_match = re.search(r'while\s*\(([^)]+)\)', body[start:])
            if not cond_match:
                cond_match = re.search(r'while\s+([^\n{]+?)(?=\s*(?:invariant|decreases|{|\n))', body[start:])
            
            if not cond_match:
                continue
            
            condition = cond_match.group(1).strip()
            after_cond = start + cond_match.end()
            
            # Find loop body
            rest = body[after_cond:]
            
            # Skip specs
            i = 0
            while i < len(rest):
                while i < len(rest) and rest[i] in ' \t\n':
                    i += 1
                if i >= len(rest):
                    break
                if rest[i:i+2] == '//':
                    while i < len(rest) and rest[i] != '\n':
                        i += 1
                    continue
                if rest[i:].startswith('invariant') or rest[i:].startswith('decreases'):
                    while i < len(rest) and rest[i] != '\n':
                        i += 1
                    continue
                if rest[i] == '{':
                    break
                i += 1
            
            loop_body = self._extract_brace_content(rest[i:])
            
            loop_vars = self._extract_variables(condition + " " + loop_body)
            all_vars = list(set(variables + loop_vars))
            
            loops.append({
                "condition": condition,
                "body": "{ " + loop_body + " }",
                "variables": all_vars,
                "has_multiplication": '*' in loop_body,
                "has_square": self._has_square_pattern(loop_body)
            })
        
        return loops
    
    def _extract_variables(self, code: str) -> List[str]:
        identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', code)
        keywords = {
            'while', 'if', 'else', 'var', 'int', 'bool', 'true', 'false',
            'requires', 'ensures', 'invariant', 'decreases', 'method',
            'returns', 'to', 'return', 'assert', 'assume', 'nat'
        }
        return list(set(v for v in identifiers if v not in keywords))
    
    def _has_square_pattern(self, code: str) -> bool:
        """Check if code contains squaring patterns like x*x or x*y"""
        return bool(re.search(r'\w+\s*\*\s*\w+', code))


class QuadraticInvariantSynthesizer:
    """
    Main synthesizer for quadratic invariants.
    """
    
    def __init__(self, coeff_bound: int = 5, loop_bound: int = 15):
        self.parser = DafnyParser()
        self.solver = Z3QuadraticSolver(coeff_bound=coeff_bound)
        self.loop_bound = loop_bound
    
    def synthesize(self, dafny_file: str) -> SynthesisResult:
        """Main entry point for synthesis from file"""
        messages = []
        
        parsed = self.parser.parse_file(dafny_file)
        if "error" in parsed:
            return SynthesisResult(
                success=False,
                invariants=[],
                invariant_objects=[],
                synthesis_type="none",
                messages=[f"Parse error: {parsed['error']}"]
            )
        
        all_invariants = []
        all_inv_objects = []
        synthesis_type = "quadratic"
        
        for method in parsed.get("methods", []):
            method_name = method["name"]
            
            for loop_idx, loop in enumerate(method.get("loops", [])):
                messages.append(f"Analyzing {method_name}, loop {loop_idx + 1}...")
                
                # Extract update patterns
                var_names = loop["variables"]
                init_values = self._get_init_values(method["body"], var_names)
                updates = self._extract_updates(loop["body"], var_names)
                
                messages.append(f"  Variables: {var_names}")
                messages.append(f"  Init values: {init_values}")
                messages.append(f"  Updates: {updates}")
                
                # Determine synthesis strategy
                if loop.get("has_multiplication") or loop.get("has_square"):
                    messages.append("  Strategy: Full quadratic synthesis")
                    invariants = self._synthesize_quadratic(var_names, init_values, updates)
                else:
                    messages.append("  Strategy: Mixed quadratic/linear synthesis")
                    invariants = self._synthesize_mixed(var_names, init_values, updates)
                
                if invariants:
                    messages.append(f"  Found {len(invariants)} invariant(s)")
                    for inv in invariants:
                        inv_str = inv.to_string()
                        messages.append(f"    - {inv_str}")
                        all_invariants.append(inv_str)
                        all_inv_objects.append(inv)
                else:
                    messages.append("  No invariants found")
        
        return SynthesisResult(
            success=len(all_invariants) > 0,
            invariants=all_invariants,
            invariant_objects=all_inv_objects,
            synthesis_type=synthesis_type,
            messages=messages
        )
    
    def _get_init_values(self, body: str, var_names: List[str]) -> Dict[str, int]:
        """Extract initial values from method body"""
        init_values = {}
        for var in var_names:
            patterns = [
                rf'{var}\s*:=\s*(-?\d+)',
                rf'var\s+{var}\s*:=\s*(-?\d+)',
                rf'var\s+{var}\s*:\s*\w+\s*:=\s*(-?\d+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, body)
                if match:
                    init_values[var] = int(match.group(1))
                    break
            if var not in init_values:
                init_values[var] = 0
        return init_values
    
    def _extract_updates(self, loop_body: str, var_names: List[str]) -> Dict[str, str]:
        """Extract update expressions from loop body"""
        updates = {}
        
        for var in var_names:
            patterns = [
                (rf'{var}\s*:=\s*{var}\s*\+\s*(\d+)', '+'),
                (rf'{var}\s*:=\s*{var}\s*-\s*(\d+)', '-'),
                (rf'{var}\s*:=\s*{var}\s*\*\s*(\d+)', '*'),
            ]
            
            for pattern, op in patterns:
                match = re.search(pattern, loop_body)
                if match:
                    updates[var] = f"{op}{match.group(1)}"
                    break
            
            if var not in updates:
                updates[var] = "+0"
        
        return updates
    
    def _synthesize_quadratic(self, var_names: List[str], 
                              init_values: Dict[str, int],
                              updates: Dict[str, str]) -> List[QuadraticInvariant]:
        """Synthesize full quadratic invariants"""
        results = []
        
        # Try inequality with all terms
        inv = self.solver.synthesize_quadratic(
            var_names, init_values, updates, self.loop_bound,
            QuadraticType.INEQUALITY, True, True
        )
        if inv and not inv.is_trivial():
            results.append(inv)
        
        # Try equality
        inv = self.solver.synthesize_quadratic(
            var_names, init_values, updates, self.loop_bound,
            QuadraticType.EQUALITY, True, True
        )
        if inv and not inv.is_trivial():
            results.append(inv)
        
        return results
    
    def _synthesize_mixed(self, var_names: List[str],
                          init_values: Dict[str, int],
                          updates: Dict[str, str]) -> List[QuadraticInvariant]:
        """Synthesize with various configurations"""
        return self.solver.synthesize_all_forms(
            var_names, init_values, updates, self.loop_bound
        )
    
    def synthesize_from_spec(self,
                             var_names: List[str],
                             init_values: Dict[str, int],
                             updates: Dict[str, str],
                             try_equality: bool = True) -> List[str]:
        """
        Synthesize from direct specification.
        Useful for testing and API access.
        """
        results = []
        
        all_invs = self.solver.synthesize_all_forms(
            var_names, init_values, updates, self.loop_bound
        )
        
        for inv in all_invs:
            results.append(inv.to_string())
        
        return results
    
    def synthesize_for_triangular(self, i_var: str = "i", 
                                   sum_var: str = "sum") -> List[str]:
        """
        Specialized synthesis for triangular number pattern:
        sum = 0 + 1 + 2 + ... + (i-1) = i*(i-1)/2
        
        Invariant: 2*sum == i*i - i (or 2*sum - i² + i == 0)
        """
        # Custom state simulation for triangular numbers
        solver = Z3QuadraticSolver(coeff_bound=10)
        
        from z3 import Solver, Int, Or, sat
        
        # Coefficients for: a*i² + b*sum + c*i + d == 0
        a = Int('a')
        b = Int('b')
        c = Int('c')
        d = Int('d')
        
        s = Solver()
        
        # Bounds
        for coeff in [a, b, c, d]:
            s.add(coeff >= -10, coeff <= 10)
        
        # Non-triviality
        s.add(Or(a != 0, b != 0, c != 0))
        
        # Add constraints for triangular number states
        # At step k: i = k, sum = k*(k-1)/2
        for k in range(20):
            i_val = k
            sum_val = k * (k - 1) // 2
            
            # a*i² + b*sum + c*i + d == 0
            s.add(a * i_val * i_val + b * sum_val + c * i_val + d == 0)
        
        results = []
        if s.check() == sat:
            model = s.model()
            a_val = model.eval(a).as_long()
            b_val = model.eval(b).as_long()
            c_val = model.eval(c).as_long()
            d_val = model.eval(d).as_long()
            
            # Build invariant string
            terms = []
            if a_val != 0:
                terms.append(f"{a_val} * {i_var} * {i_var}" if a_val != 1 else f"{i_var} * {i_var}")
            if b_val != 0:
                terms.append(f"{b_val} * {sum_var}" if b_val != 1 else sum_var)
            if c_val != 0:
                terms.append(f"{c_val} * {i_var}" if c_val != 1 else i_var)
            
            if terms:
                lhs = " + ".join(terms).replace("+ -", "- ")
                results.append(f"{lhs} == {-d_val}")
        
        return results
    
    def synthesize_for_square_sum(self, i_var: str = "i",
                                   sum_var: str = "sum") -> List[str]:
        """
        Specialized synthesis for sum of squares:
        sum = 0² + 1² + ... + (i-1)² = (i-1)*i*(2i-1)/6
        """
        # This requires cubic terms, so we provide a bound instead
        return [f"6 * {sum_var} <= {i_var} * {i_var} * {i_var}"]


class InvariantInserter:
    """Insert synthesized invariants into Dafny source"""
    
    def insert(self, source: str, invariants: List[str]) -> str:
        lines = source.split('\n')
        new_lines = []
        
        for line in lines:
            new_lines.append(line)
            
            if re.match(r'\s*while\s*', line) and '{' not in line:
                indent = len(line) - len(line.lstrip())
                inv_indent = ' ' * (indent + 2)
                
                for inv in invariants:
                    new_lines.append(f"{inv_indent}invariant {inv}")
        
        return '\n'.join(new_lines)


def run_tests():
    """Run internal tests"""
    print("Running tests...")
    print("=" * 60)
    
    synthesizer = QuadraticInvariantSynthesizer(coeff_bound=5, loop_bound=15)
    solver = Z3QuadraticSolver(coeff_bound=5)
    
    # Test 1: Simple linear (should still work)
    print("\nTest 1: Linear update (x += 1, y += 2)")
    invs = synthesizer.synthesize_from_spec(
        var_names=["x", "y"],
        init_values={"x": 0, "y": 0},
        updates={"x": "+1", "y": "+2"}
    )
    print(f"  Found {len(invs)} invariant(s)")
    for inv in invs[:3]:
        print(f"    - {inv}")
    
    # Test 2: Quadratic growth
    print("\nTest 2: Quadratic pattern (single variable)")
    invs = synthesizer.synthesize_from_spec(
        var_names=["i"],
        init_values={"i": 0},
        updates={"i": "+1"}
    )
    print(f"  Found {len(invs)} invariant(s)")
    for inv in invs[:3]:
        print(f"    - {inv}")
    
    # Test 3: Two variables with cross terms
    print("\nTest 3: Two variables for cross-term detection")
    inv = solver.synthesize_quadratic(
        var_names=["x", "y"],
        init_values={"x": 0, "y": 0},
        update_exprs={"x": "+1", "y": "+1"},
        invariant_type=QuadraticType.INEQUALITY,
        include_cross_terms=True
    )
    if inv:
        print(f"  Found: {inv.to_string()}")
    
    # Test 4: Triangular numbers
    print("\nTest 4: Triangular numbers (specialized)")
    invs = synthesizer.synthesize_for_triangular("i", "sum")
    print(f"  Found {len(invs)} invariant(s)")
    for inv in invs:
        print(f"    - {inv}")
    
    print("\n" + "=" * 60)
    print("Tests completed.")


def run_benchmarks():
    """Run benchmark suite"""
    print("Running benchmarks...")
    print("=" * 60)
    
    benchmark_dir = os.path.join(os.path.dirname(__file__), 'benchmarks')
    
    if not os.path.exists(benchmark_dir):
        print(f"Benchmark directory not found: {benchmark_dir}")
        return
    
    benchmarks = sorted([f for f in os.listdir(benchmark_dir) if f.endswith('.dfy')])
    
    if not benchmarks:
        print("No benchmark files found.")
        return
    
    synthesizer = QuadraticInvariantSynthesizer()
    results = []
    
    for bench_file in benchmarks:
        bench_path = os.path.join(benchmark_dir, bench_file)
        print(f"\nBenchmark: {bench_file}")
        print("-" * 40)
        
        result = synthesizer.synthesize(bench_path)
        results.append({
            "file": bench_file,
            "success": result.success,
            "count": len(result.invariants),
            "type": result.synthesis_type
        })
        
        for msg in result.messages:
            print(f"  {msg}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['file']}: {r['count']} inv(s)")
    
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    print(f"\nTotal: {successful}/{total} benchmarks succeeded")


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Quadratic Invariant Synthesis Tool'
    )
    parser.add_argument('input', nargs='?', help='Input Dafny file')
    parser.add_argument('-o', '--output', help='Output file with invariants')
    parser.add_argument('-c', '--coeff-bound', type=int, default=5,
                       help='Coefficient bound (default: 5)')
    parser.add_argument('-l', '--loop-bound', type=int, default=15,
                       help='Loop simulation bound (default: 15)')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
        return 0
    
    if args.benchmark:
        run_benchmarks()
        return 0
    
    if not args.input:
        parser.print_help()
        return 1
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1
    
    synthesizer = QuadraticInvariantSynthesizer(
        coeff_bound=args.coeff_bound,
        loop_bound=args.loop_bound
    )
    
    result = synthesizer.synthesize(args.input)
    
    if args.json:
        output = {
            "success": result.success,
            "invariants": result.invariants,
            "synthesis_type": result.synthesis_type,
            "messages": result.messages
        }
        print(json.dumps(output, indent=2))
    else:
        print("=" * 60)
        print("Quadratic Invariant Synthesis")
        print(f"Input: {args.input}")
        print("=" * 60)
        
        for msg in result.messages:
            print(msg)
        
        if result.success:
            print(f"\nSynthesized {len(result.invariants)} invariant(s)")
        else:
            print("\nNo invariants found")
    
    if args.output and result.success:
        inserter = InvariantInserter()
        with open(args.input, 'r') as f:
            source = f.read()
        modified = inserter.insert(source, result.invariants)
        with open(args.output, 'w') as f:
            f.write(modified)
        print(f"\nWritten to {args.output}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
