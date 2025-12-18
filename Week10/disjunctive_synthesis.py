"""
Disjunctive Invariant Synthesis Tool
Synthesizes disjunctive invariants for programs with multiple execution paths.

Key features:
- Path analysis for conditional branches in loops
- Disjunctive invariants: inv1 || inv2 || ...
- Path-sensitive invariants: (cond1 ==> inv1) && (cond2 ==> inv2)
- Support for alternating loop behaviors
"""

import sys
import os
import re
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from path_analyzer import PathAnalyzer, LoopPathAnalysis, PathType, ExecutionPath
from z3_disjunctive_solver import (
    Z3DisjunctiveSolver, DisjunctiveInvariant, DisjunctInvariant
)


@dataclass
class SynthesisResult:
    """Result of disjunctive invariant synthesis"""
    success: bool
    invariants: List[str]
    path_analysis: Optional[LoopPathAnalysis]
    synthesis_type: str  # "linear", "disjunctive", "path_sensitive"
    messages: List[str]


class DafnyParser:
    """Simplified Dafny parser for extracting loop information"""
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse a Dafny file and extract relevant information"""
        try:
            with open(filepath, 'r') as f:
                source = f.read()
            return self.parse_source(source)
        except Exception as e:
            return {"error": str(e)}
    
    def parse_source(self, source: str) -> Dict[str, Any]:
        """Parse Dafny source and extract method/loop info"""
        result = {
            "methods": [],
            "loops": [],
            "preconditions": [],
            "postconditions": []
        }
        
        # Find method declarations
        method_start_pattern = r'method\s+(\w+)\s*\(([^)]*)\)(?:\s*returns\s*\(([^)]*)\))?'
        
        for match in re.finditer(method_start_pattern, source):
            method_name = match.group(1)
            params_str = match.group(2)
            returns_str = match.group(3) or ""
            
            # Parse parameters
            params = self._parse_params(params_str)
            returns = self._parse_params(returns_str)
            
            # Find the method body by looking for opening brace and matching it
            after_match = source[match.end():]
            
            # Find specs (requires/ensures) before the body
            specs_pattern = r'^((?:\s*requires[^\n]*\n|\s*ensures[^\n]*\n)*)'
            specs_match = re.match(specs_pattern, after_match)
            specs_str = ""
            body_start_offset = 0
            if specs_match:
                specs_str = specs_match.group(1)
                body_start_offset = specs_match.end()
            
            # Parse requires/ensures
            preconditions = re.findall(r'requires\s+([^\n]+)', specs_str)
            postconditions = re.findall(r'ensures\s+([^\n]+)', specs_str)
            preconditions = [p.strip() for p in preconditions]
            postconditions = [p.strip() for p in postconditions]
            
            # Find the body by matching braces
            rest = after_match[body_start_offset:]
            brace_start = rest.find('{')
            if brace_start == -1:
                continue
            
            # Match braces to find complete body
            brace_count = 0
            body = ""
            started = False
            for i, c in enumerate(rest):
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
                    body += c
            
            # Find loops in body
            loops = self._extract_loops(body, params + returns)
            
            method_data = {
                "name": method_name,
                "parameters": params,
                "returns": returns,
                "preconditions": preconditions,
                "postconditions": postconditions,
                "body": body,
                "loops": loops
            }
            
            result["methods"].append(method_data)
            result["loops"].extend(loops)
            result["preconditions"].extend(preconditions)
            result["postconditions"].extend(postconditions)
        
        return result
    
    def _parse_params(self, params_str: str) -> List[str]:
        """Parse parameter string into list of variable names"""
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
    
    def _extract_loops(self, body: str, variables: List[str]) -> List[Dict]:
        """Extract while loops from method body"""
        loops = []
        
        # Find while loop start positions
        while_starts = [m.start() for m in re.finditer(r'\bwhile\b', body)]
        
        for start in while_starts:
            # Extract condition - Dafny allows both `while (cond)` and `while cond`
            cond_match = re.search(r'while\s*\(([^)]+)\)', body[start:])
            if not cond_match:
                # Try without parentheses
                cond_match = re.search(r'while\s+([^\n{]+?)(?=\s*(?:invariant|decreases|{|\n))', body[start:])
            
            if not cond_match:
                continue
            
            condition = cond_match.group(1).strip()
            after_cond = start + cond_match.end()
            
            # Skip past any invariant/decreases/comment lines
            rest = body[after_cond:]
            
            # Find opening brace, skipping comments and specs
            brace_idx = -1
            i = 0
            while i < len(rest):
                # Skip whitespace
                while i < len(rest) and rest[i] in ' \t\n':
                    i += 1
                if i >= len(rest):
                    break
                
                # Skip comments
                if rest[i:i+2] == '//':
                    while i < len(rest) and rest[i] != '\n':
                        i += 1
                    continue
                
                # Skip invariant/decreases
                if rest[i:].startswith('invariant') or rest[i:].startswith('decreases'):
                    while i < len(rest) and rest[i] != '\n':
                        i += 1
                    continue
                
                # Found brace
                if rest[i] == '{':
                    brace_idx = i
                    break
                
                i += 1
            
            if brace_idx == -1:
                continue
            
            # Extract specs before brace
            specs = rest[:brace_idx]
            
            # Match braces to find loop body
            rest = rest[brace_idx:]
            brace_count = 0
            loop_body = ""
            in_body = False
            
            for c in rest:
                if c == '{':
                    brace_count += 1
                    if not in_body:
                        in_body = True
                        continue
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        break
                
                if in_body:
                    loop_body += c
            
            # Extract existing invariants from specs
            invariants = re.findall(r'invariant\s+([^\n]+)', specs)
            invariants = [inv.strip() for inv in invariants]
            
            # Find variables used in loop
            loop_vars = self._extract_variables(condition + " " + loop_body)
            all_vars = list(set(variables + loop_vars))
            
            # Detect if loop has conditionals
            has_conditionals = bool(re.search(r'\bif\b', loop_body))
            
            loops.append({
                "condition": condition,
                "body": "{ " + loop_body + " }",
                "variables": all_vars,
                "existing_invariants": invariants,
                "has_conditionals": has_conditionals
            })
        
        return loops
    
    def _extract_variables(self, code: str) -> List[str]:
        """Extract variable names from code"""
        # Find all identifiers
        identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', code)
        
        # Filter out keywords
        keywords = {
            'while', 'if', 'else', 'var', 'int', 'bool', 'true', 'false',
            'requires', 'ensures', 'invariant', 'decreases', 'method',
            'returns', 'to', 'return', 'assert', 'assume'
        }
        
        return list(set(v for v in identifiers if v not in keywords))


class DisjunctiveInvariantSynthesizer:
    """
    Main synthesizer for disjunctive invariants.
    """
    
    def __init__(self, coeff_bound: int = 10, max_disjuncts: int = 3):
        self.parser = DafnyParser()
        self.path_analyzer = PathAnalyzer()
        self.solver = Z3DisjunctiveSolver(coeff_bound=coeff_bound)
        self.max_disjuncts = max_disjuncts
    
    def synthesize(self, dafny_file: str) -> SynthesisResult:
        """Main entry point for synthesis"""
        messages = []
        
        # Parse file
        parsed = self.parser.parse_file(dafny_file)
        if "error" in parsed:
            return SynthesisResult(
                success=False,
                invariants=[],
                path_analysis=None,
                synthesis_type="none",
                messages=[f"Parse error: {parsed['error']}"]
            )
        
        all_invariants = []
        all_analyses = []
        synthesis_type = "linear"
        
        for method in parsed.get("methods", []):
            method_name = method["name"]
            
            for loop_idx, loop in enumerate(method.get("loops", [])):
                messages.append(f"Analyzing {method_name}, loop {loop_idx + 1}...")
                
                # Analyze paths in the loop
                analysis = self.path_analyzer.analyze_loop(
                    loop["condition"],
                    loop["body"],
                    loop["variables"]
                )
                all_analyses.append(analysis)
                
                messages.append(f"  Path type: {analysis.path_type.value}")
                messages.append(f"  Paths found: {len(analysis.paths)}")
                
                # Determine synthesis strategy
                if analysis.requires_disjunctive_invariant():
                    messages.append("  Strategy: Disjunctive synthesis")
                    synthesis_type = "disjunctive"
                    invariants = self._synthesize_disjunctive(loop, analysis)
                elif len(analysis.paths) > 1:
                    messages.append("  Strategy: Path-sensitive synthesis")
                    synthesis_type = "path_sensitive"
                    invariants = self._synthesize_path_sensitive(loop, analysis)
                else:
                    messages.append("  Strategy: Linear synthesis")
                    invariants = self._synthesize_linear(loop, analysis)
                
                if invariants:
                    messages.append(f"  Found {len(invariants)} invariant(s)")
                    for inv in invariants:
                        messages.append(f"    - {inv}")
                        all_invariants.append(inv)
                else:
                    messages.append("  No invariants found")
        
        return SynthesisResult(
            success=len(all_invariants) > 0,
            invariants=all_invariants,
            path_analysis=all_analyses[0] if all_analyses else None,
            synthesis_type=synthesis_type,
            messages=messages
        )
    
    def _get_init_values(self, loop: Dict) -> Dict[str, int]:
        """Extract initial values from context before loop"""
        init_values = {}
        for var in loop["variables"]:
            # Look for initialization patterns
            # Default to 0 for now
            init_values[var] = 0
        return init_values
    
    def _synthesize_linear(self, loop: Dict, 
                          analysis: LoopPathAnalysis) -> List[str]:
        """Synthesize simple linear invariants"""
        var_names = list(analysis.all_variables)
        init_values = self._get_init_values(loop)
        
        results = []
        inv = self.solver.synthesize_disjunctive(
            var_names, init_values, analysis, num_disjuncts=1
        )
        if inv:
            results.append(inv.to_string())
        
        return results
    
    def _synthesize_disjunctive(self, loop: Dict,
                                analysis: LoopPathAnalysis) -> List[str]:
        """Synthesize disjunctive invariants"""
        var_names = list(analysis.all_variables)
        init_values = self._get_init_values(loop)
        
        results = []
        
        # Try different numbers of disjuncts
        for n in range(2, self.max_disjuncts + 1):
            inv = self.solver.synthesize_disjunctive(
                var_names, init_values, analysis, num_disjuncts=n
            )
            if inv:
                results.append(inv.to_string())
        
        # Also try path-sensitive
        inv_ps = self.solver.synthesize_path_sensitive(
            var_names, init_values, analysis
        )
        if inv_ps:
            results.append(inv_ps.to_conjunctive_string())
        
        return results
    
    def _synthesize_path_sensitive(self, loop: Dict,
                                   analysis: LoopPathAnalysis) -> List[str]:
        """Synthesize path-sensitive invariants"""
        var_names = list(analysis.all_variables)
        init_values = self._get_init_values(loop)
        
        results = []
        
        inv = self.solver.synthesize_path_sensitive(
            var_names, init_values, analysis
        )
        if inv:
            results.append(inv.to_conjunctive_string())
        
        # Also try simple disjunctive
        inv_disj = self.solver.synthesize_disjunctive(
            var_names, init_values, analysis, num_disjuncts=len(analysis.paths)
        )
        if inv_disj:
            results.append(inv_disj.to_string())
        
        return results
    
    def synthesize_from_spec(self, 
                             var_names: List[str],
                             init_values: Dict[str, int],
                             loop_body: str,
                             loop_condition: str = "true") -> List[str]:
        """
        Synthesize from a direct specification.
        Useful for testing and API access.
        """
        analysis = self.path_analyzer.analyze_loop(
            loop_condition, loop_body, var_names
        )
        
        results = []
        
        # Try all synthesis strategies
        all_invs = self.solver.synthesize_all_forms(
            var_names, init_values, analysis, self.max_disjuncts
        )
        
        for inv in all_invs:
            if inv.is_path_sensitive:
                results.append(inv.to_conjunctive_string())
            else:
                results.append(inv.to_string())
        
        return results


class InvariantInserter:
    """Insert synthesized invariants into Dafny source"""
    
    def insert(self, source: str, invariants: List[str]) -> str:
        """Insert invariants after while loop declarations"""
        lines = source.split('\n')
        new_lines = []
        
        for line in lines:
            new_lines.append(line)
            
            # Check for while loop without existing invariants
            if re.match(r'\s*while\s*\(', line) and '{' not in line:
                indent = len(line) - len(line.lstrip())
                inv_indent = ' ' * (indent + 2)
                
                for inv in invariants:
                    new_lines.append(f"{inv_indent}invariant {inv}")
        
        return '\n'.join(new_lines)


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Disjunctive Invariant Synthesis Tool'
    )
    parser.add_argument('input', nargs='?', help='Input Dafny file')
    parser.add_argument('-o', '--output', help='Output file with invariants')
    parser.add_argument('-c', '--coeff-bound', type=int, default=10,
                       help='Coefficient bound (default: 10)')
    parser.add_argument('-d', '--max-disjuncts', type=int, default=3,
                       help='Max disjuncts (default: 3)')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--analyze', action='store_true', 
                       help='Only analyze paths, no synthesis')
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
    
    synthesizer = DisjunctiveInvariantSynthesizer(
        coeff_bound=args.coeff_bound,
        max_disjuncts=args.max_disjuncts
    )
    
    if args.analyze:
        # Just analyze paths
        parsed = synthesizer.parser.parse_file(args.input)
        for method in parsed.get("methods", []):
            for loop in method.get("loops", []):
                analysis = synthesizer.path_analyzer.analyze_loop(
                    loop["condition"], loop["body"], loop["variables"]
                )
                print(synthesizer.path_analyzer.summarize_analysis(analysis))
        return 0
    
    # Full synthesis
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
        print("Disjunctive Invariant Synthesis")
        print(f"Input: {args.input}")
        print("=" * 60)
        
        for msg in result.messages:
            print(msg)
        
        if result.success:
            print(f"\nSynthesized {len(result.invariants)} invariant(s)")
            print(f"Synthesis type: {result.synthesis_type}")
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


def run_tests():
    """Run internal tests"""
    print("Running tests...")
    print("=" * 60)
    
    synthesizer = DisjunctiveInvariantSynthesizer()
    
    # Test 1: Simple linear
    print("\nTest 1: Linear loop")
    invs = synthesizer.synthesize_from_spec(
        var_names=["i", "x"],
        init_values={"i": 0, "x": 0},
        loop_body="{ i := i + 1; x := x + 2; }",
        loop_condition="i < n"
    )
    print(f"  Found {len(invs)} invariant(s)")
    for inv in invs[:2]:
        print(f"    - {inv}")
    
    # Test 2: If-then-else
    print("\nTest 2: If-then-else loop")
    invs = synthesizer.synthesize_from_spec(
        var_names=["i", "x"],
        init_values={"i": 0, "x": 0},
        loop_body="""{ 
            if (i % 2 == 0) { x := x + 2; } 
            else { x := x + 1; }
            i := i + 1;
        }""",
        loop_condition="i < n"
    )
    print(f"  Found {len(invs)} invariant(s)")
    for inv in invs[:3]:
        print(f"    - {inv}")
    
    # Test 3: Multiple paths
    print("\nTest 3: Multiple conditional paths")
    invs = synthesizer.synthesize_from_spec(
        var_names=["i", "x", "y"],
        init_values={"i": 0, "x": 0, "y": 0},
        loop_body="""{ 
            if (x > 0) { y := y + 1; }
            x := x - 1;
            i := i + 1;
        }""",
        loop_condition="i < n"
    )
    print(f"  Found {len(invs)} invariant(s)")
    for inv in invs[:3]:
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
    
    synthesizer = DisjunctiveInvariantSynthesizer()
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
        print(f"  {status} {r['file']}: {r['count']} inv(s), type={r['type']}")
    
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    print(f"\nTotal: {successful}/{total} benchmarks succeeded")


if __name__ == "__main__":
    sys.exit(main())
