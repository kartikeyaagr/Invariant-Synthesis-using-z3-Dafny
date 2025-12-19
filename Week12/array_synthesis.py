"""
Array Invariant Synthesis Tool
Synthesizes loop invariants for programs involving arrays and lists.

Supports:
- Array bounds invariants
- Quantified invariants (forall/exists)
- Accumulator patterns
- Copy/transform patterns
- Sorted array invariants
"""

import sys
import os
import re
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from array_analyzer import (
    ArrayAnalyzer, LoopArrayAnalysis, ArrayPattern,
    ArrayInvariantTemplates, QuantifiedInvariant
)
from z3_array_solver import (
    Z3ArraySolver, ArrayInvariant, QuantifiedInvariantSynthesizer
)


@dataclass
class SynthesisResult:
    """Result of array invariant synthesis"""
    success: bool
    invariants: List[str]
    invariant_details: List[ArrayInvariant]
    pattern: str
    messages: List[str]


class DafnyArrayParser:
    """Parser for Dafny programs with array operations"""
    
    def __init__(self):
        self.array_type_pattern = re.compile(r'(\w+)\s*:\s*array<(\w+)>')
        self.seq_type_pattern = re.compile(r'(\w+)\s*:\s*seq<(\w+)>')
    
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
            "arrays": [],
            "loops": []
        }
        
        # Find method declarations
        method_pattern = r'method\s+(\w+)\s*\(([^)]*)\)(?:\s*returns\s*\(([^)]*)\))?'
        
        for match in re.finditer(method_pattern, source):
            method_name = match.group(1)
            params_str = match.group(2)
            returns_str = match.group(3) or ""
            
            # Parse parameters and find arrays
            params, arrays = self._parse_params_with_arrays(params_str)
            returns, ret_arrays = self._parse_params_with_arrays(returns_str)
            arrays.extend(ret_arrays)
            
            # Find method body
            after_match = source[match.end():]
            body = self._extract_body(after_match)
            
            # Find arrays declared in body
            body_arrays = self._find_body_arrays(body)
            arrays.extend(body_arrays)
            
            # Extract loops
            loops = self._extract_loops(body, params + returns, arrays)
            
            # Parse pre/postconditions
            specs = self._extract_specs(after_match)
            
            method_data = {
                "name": method_name,
                "parameters": params,
                "returns": returns,
                "arrays": list(set(arrays)),
                "body": body,
                "loops": loops,
                "preconditions": specs["requires"],
                "postconditions": specs["ensures"],
                "modifies": specs.get("modifies", [])
            }
            
            result["methods"].append(method_data)
            result["arrays"].extend(method_data["arrays"])
            result["loops"].extend(loops)
        
        result["arrays"] = list(set(result["arrays"]))
        return result
    
    def _parse_params_with_arrays(self, params_str: str) -> Tuple[List[str], List[str]]:
        """Parse parameters and identify array types"""
        if not params_str.strip():
            return [], []
        
        params = []
        arrays = []
        
        for part in params_str.split(','):
            part = part.strip()
            if ':' in part:
                name = part.split(':')[0].strip()
                type_part = part.split(':')[1].strip()
                params.append(name)
                
                if 'array' in type_part.lower() or 'seq' in type_part.lower():
                    arrays.append(name)
            elif part:
                params.append(part)
        
        return params, arrays
    
    def _extract_body(self, text: str) -> str:
        """Extract method body"""
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
    
    def _find_body_arrays(self, body: str) -> List[str]:
        """Find array declarations in method body"""
        arrays = []
        
        # var a := new int[n]
        new_array = re.findall(r'var\s+(\w+)\s*:=\s*new\s+\w+\s*\[', body)
        arrays.extend(new_array)
        
        # var a: array<int>
        typed_array = re.findall(r'var\s+(\w+)\s*:\s*array<\w+>', body)
        arrays.extend(typed_array)
        
        return arrays
    
    def _extract_specs(self, text: str) -> Dict[str, List[str]]:
        """Extract requires/ensures/modifies clauses"""
        specs = {
            "requires": [],
            "ensures": [],
            "modifies": []
        }
        
        # Find all specs before the body
        brace_pos = text.find('{')
        if brace_pos > 0:
            specs_text = text[:brace_pos]
            
            specs["requires"] = re.findall(r'requires\s+([^\n]+)', specs_text)
            specs["ensures"] = re.findall(r'ensures\s+([^\n]+)', specs_text)
            specs["modifies"] = re.findall(r'modifies\s+([^\n]+)', specs_text)
        
        return specs
    
    def _extract_loops(self, body: str, variables: List[str], 
                       arrays: List[str]) -> List[Dict]:
        """Extract while loops with array operations"""
        loops = []
        
        while_starts = [m.start() for m in re.finditer(r'\bwhile\b', body)]
        
        for start in while_starts:
            # Extract condition
            cond_match = re.search(r'while\s*\(([^)]+)\)', body[start:])
            if not cond_match:
                cond_match = re.search(r'while\s+([^\n{]+?)(?=\s*(?:invariant|decreases|{|\n))', 
                                      body[start:])
            if not cond_match:
                continue
            
            condition = cond_match.group(1).strip()
            after_cond = start + cond_match.end()
            
            # Extract existing invariants
            rest = body[after_cond:]
            existing_invariants = []
            
            # Find loop body
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
                if rest[i:].startswith('invariant'):
                    end = rest.find('\n', i)
                    if end > i:
                        inv_text = rest[i+9:end].strip()
                        existing_invariants.append(inv_text)
                        i = end
                    continue
                if rest[i:].startswith('decreases'):
                    end = rest.find('\n', i)
                    i = end if end > i else i + 1
                    continue
                if rest[i] == '{':
                    break
                i += 1
            
            # Extract loop body
            loop_body = self._extract_body(rest[i:])
            
            # Find arrays used in this loop
            loop_arrays = []
            for arr in arrays:
                if f'{arr}[' in loop_body or f'{arr}.Length' in condition + loop_body:
                    loop_arrays.append(arr)
            
            # Find all variables in loop
            loop_vars = self._extract_variables(condition + " " + loop_body)
            all_vars = list(set(variables + loop_vars))
            
            loops.append({
                "condition": condition,
                "body": "{ " + loop_body + " }",
                "variables": all_vars,
                "arrays": loop_arrays,
                "existing_invariants": existing_invariants,
                "has_array_access": bool(loop_arrays)
            })
        
        return loops
    
    def _extract_variables(self, code: str) -> List[str]:
        """Extract variable names from code"""
        identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', code)
        keywords = {
            'while', 'if', 'else', 'var', 'int', 'bool', 'true', 'false',
            'requires', 'ensures', 'invariant', 'decreases', 'method',
            'returns', 'to', 'return', 'assert', 'assume', 'nat', 'array',
            'seq', 'new', 'forall', 'exists', 'old', 'fresh', 'modifies',
            'Length', 'multiset'
        }
        return list(set(v for v in identifiers if v not in keywords))


class ArrayInvariantSynthesizer:
    """
    Main synthesizer for array invariants.
    """
    
    def __init__(self, coeff_bound: int = 10):
        self.parser = DafnyArrayParser()
        self.analyzer = ArrayAnalyzer()
        self.solver = Z3ArraySolver(coeff_bound=coeff_bound)
        self.quant_synth = QuantifiedInvariantSynthesizer()
    
    def synthesize(self, dafny_file: str) -> SynthesisResult:
        """Main entry point for synthesis from file"""
        messages = []
        
        parsed = self.parser.parse_file(dafny_file)
        if "error" in parsed:
            return SynthesisResult(
                success=False,
                invariants=[],
                invariant_details=[],
                pattern="none",
                messages=[f"Parse error: {parsed['error']}"]
            )
        
        all_invariants = []
        all_details = []
        detected_pattern = "none"
        
        for method in parsed.get("methods", []):
            method_name = method["name"]
            method_arrays = method.get("arrays", [])
            
            for loop_idx, loop in enumerate(method.get("loops", [])):
                messages.append(f"Analyzing {method_name}, loop {loop_idx + 1}...")
                
                if not loop.get("has_array_access"):
                    messages.append("  No array access detected, skipping...")
                    continue
                
                # Analyze the loop
                analysis = self.analyzer.analyze_loop(
                    loop["condition"],
                    loop["body"],
                    loop["variables"],
                    loop.get("arrays", method_arrays)
                )
                
                detected_pattern = analysis.pattern.value
                messages.append(f"  Pattern: {detected_pattern}")
                messages.append(f"  Arrays: {list(analysis.arrays.keys())}")
                messages.append(f"  Index: {analysis.index_var}")
                
                # Synthesize invariants
                invariants = self.solver.synthesize_all(
                    analysis, loop["variables"]
                )
                
                # Add pattern-specific quantified invariants
                quant_invs = self._synthesize_pattern_invariants(analysis, loop)
                for inv_str in quant_invs:
                    invariants.append(ArrayInvariant(
                        invariant_type="quantified",
                        expression=inv_str,
                        confidence=0.75
                    ))
                
                if invariants:
                    messages.append(f"  Found {len(invariants)} invariant(s)")
                    for inv in invariants:
                        messages.append(f"    [{inv.confidence:.1f}] {inv.expression}")
                        all_invariants.append(inv.expression)
                        all_details.append(inv)
                else:
                    messages.append("  No invariants found")
        
        return SynthesisResult(
            success=len(all_invariants) > 0,
            invariants=all_invariants,
            invariant_details=all_details,
            pattern=detected_pattern,
            messages=messages
        )
    
    def _synthesize_pattern_invariants(self, analysis: LoopArrayAnalysis,
                                       loop: Dict) -> List[str]:
        """Synthesize pattern-specific quantified invariants"""
        results = []
        
        idx = analysis.index_var
        
        if analysis.pattern == ArrayPattern.INIT:
            # Initialization: forall k :: 0 <= k < i ==> a[k] == val
            for arr in analysis.modified_arrays:
                for access in analysis.accesses:
                    if access.array == arr and access.value:
                        val = access.value
                        results.append(
                            f"forall k :: 0 <= k < {idx} ==> {arr}[k] == {val}"
                        )
        
        elif analysis.pattern == ArrayPattern.COPY:
            # Copy: forall k :: 0 <= k < i ==> dst[k] == src[k]
            src = list(analysis.read_arrays - analysis.modified_arrays)
            dst = list(analysis.modified_arrays)
            if src and dst:
                results.append(
                    f"forall k :: 0 <= k < {idx} ==> {dst[0]}[k] == {src[0]}[k]"
                )
        
        elif analysis.pattern == ArrayPattern.TRANSFORM:
            # Transform: some property holds for processed elements
            for arr in analysis.modified_arrays:
                results.append(
                    f"forall k :: 0 <= k < {idx} ==> {arr}[k] >= 0"
                )
        
        elif analysis.pattern == ArrayPattern.ACCUMULATE:
            # Accumulator: partial sum/product
            for arr in analysis.read_arrays:
                results.append(f"0 <= {idx} <= {arr}.Length")
        
        elif analysis.pattern == ArrayPattern.LINEAR_SCAN:
            # Linear scan: basic bounds
            for arr in analysis.read_arrays | analysis.modified_arrays:
                results.append(f"0 <= {idx} <= {arr}.Length")
        
        return results
    
    def synthesize_from_spec(self, 
                             loop_condition: str,
                             loop_body: str,
                             variables: List[str],
                             arrays: List[str]) -> List[str]:
        """
        Synthesize from direct specification.
        """
        analysis = self.analyzer.analyze_loop(
            loop_condition, loop_body, variables, arrays
        )
        
        invariants = self.solver.synthesize_all(analysis, variables)
        
        # Add pattern-specific
        quant_invs = self._synthesize_pattern_invariants(
            analysis, 
            {"condition": loop_condition, "body": loop_body}
        )
        
        return [inv.expression for inv in invariants] + quant_invs


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
    
    synthesizer = ArrayInvariantSynthesizer()
    
    # Test 1: Array initialization
    print("\nTest 1: Array initialization")
    invs = synthesizer.synthesize_from_spec(
        loop_condition="i < a.Length",
        loop_body="{ a[i] := 0; i := i + 1; }",
        variables=["i", "a"],
        arrays=["a"]
    )
    print(f"  Found {len(invs)} invariant(s)")
    for inv in invs[:5]:
        print(f"    - {inv}")
    
    # Test 2: Array sum
    print("\nTest 2: Array sum")
    invs = synthesizer.synthesize_from_spec(
        loop_condition="i < a.Length",
        loop_body="{ sum := sum + a[i]; i := i + 1; }",
        variables=["i", "sum", "a"],
        arrays=["a"]
    )
    print(f"  Found {len(invs)} invariant(s)")
    for inv in invs[:5]:
        print(f"    - {inv}")
    
    # Test 3: Array copy
    print("\nTest 3: Array copy")
    invs = synthesizer.synthesize_from_spec(
        loop_condition="i < src.Length",
        loop_body="{ dst[i] := src[i]; i := i + 1; }",
        variables=["i", "src", "dst"],
        arrays=["src", "dst"]
    )
    print(f"  Found {len(invs)} invariant(s)")
    for inv in invs[:5]:
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
    
    synthesizer = ArrayInvariantSynthesizer()
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
            "pattern": result.pattern
        })
        
        for msg in result.messages:
            print(f"  {msg}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['file']}: {r['count']} inv(s), pattern={r['pattern']}")
    
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    print(f"\nTotal: {successful}/{total} benchmarks succeeded")


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Array Invariant Synthesis Tool'
    )
    parser.add_argument('input', nargs='?', help='Input Dafny file')
    parser.add_argument('-o', '--output', help='Output file with invariants')
    parser.add_argument('-c', '--coeff-bound', type=int, default=10,
                       help='Coefficient bound (default: 10)')
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
    
    synthesizer = ArrayInvariantSynthesizer(coeff_bound=args.coeff_bound)
    result = synthesizer.synthesize(args.input)
    
    if args.json:
        output = {
            "success": result.success,
            "invariants": result.invariants,
            "pattern": result.pattern,
            "messages": result.messages
        }
        print(json.dumps(output, indent=2))
    else:
        print("=" * 60)
        print("Array Invariant Synthesis")
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
