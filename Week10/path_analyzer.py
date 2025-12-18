"""
Path Analyzer for Disjunctive Invariant Synthesis
Analyzes conditional branches within loop bodies to identify execution paths.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class PathType(Enum):
    LINEAR = "linear"           # No branches
    IF_THEN = "if_then"         # Single if without else
    IF_THEN_ELSE = "if_else"    # If with else
    MULTI_BRANCH = "multi"      # Multiple/nested conditionals


@dataclass
class VariableUpdate:
    """Represents an update to a variable"""
    variable: str
    expression: str
    delta: Optional[int] = None  # For simple x := x + delta patterns
    
    def __repr__(self):
        return f"{self.variable} := {self.expression}"


@dataclass
class ExecutionPath:
    """Represents a single execution path through a loop body"""
    path_id: int
    condition: Optional[str]       # Guard condition for this path (None if unconditional)
    negated_conditions: List[str]  # Conditions that must be false
    updates: Dict[str, VariableUpdate]
    is_else_branch: bool = False
    
    def get_path_condition(self) -> str:
        """Get the full path condition as a string"""
        parts = []
        if self.condition:
            parts.append(self.condition)
        for neg in self.negated_conditions:
            parts.append(f"!({neg})")
        
        if not parts:
            return "true"
        return " && ".join(parts)


@dataclass
class LoopPathAnalysis:
    """Complete analysis of paths through a loop"""
    loop_condition: str
    paths: List[ExecutionPath]
    all_variables: Set[str]
    path_type: PathType
    has_disjunctive_behavior: bool
    
    def get_path_count(self) -> int:
        return len(self.paths)
    
    def requires_disjunctive_invariant(self) -> bool:
        """Check if this loop likely needs a disjunctive invariant"""
        if len(self.paths) <= 1:
            return False
        
        # Check if different paths have different update patterns
        update_patterns = []
        for path in self.paths:
            pattern = frozenset(
                (v, u.delta) for v, u in path.updates.items() 
                if u.delta is not None
            )
            update_patterns.append(pattern)
        
        # If patterns differ, likely needs disjunctive invariant
        return len(set(update_patterns)) > 1


class PathAnalyzer:
    """Analyzes loop bodies to extract execution paths"""
    
    def __init__(self):
        self.path_counter = 0
    
    def analyze_loop(self, loop_condition: str, loop_body: str, 
                     variables: List[str]) -> LoopPathAnalysis:
        """
        Analyze a loop body and extract all execution paths.
        
        Args:
            loop_condition: The while loop guard condition
            loop_body: The body of the loop as a string
            variables: List of variable names in scope
        
        Returns:
            LoopPathAnalysis with all paths identified
        """
        self.path_counter = 0
        paths = []
        
        # Normalize whitespace for easier parsing
        normalized_body = ' '.join(loop_body.split())
        
        # Check for if-else pattern with proper brace matching
        if_pattern = r'\bif\s*\(?\s*([^){]+?)\s*\)?\s*\{'
        if_matches = list(re.finditer(if_pattern, normalized_body))
        
        if not if_matches:
            # No conditionals - single linear path
            path = self._extract_linear_path(loop_body, variables, None, [])
            paths.append(path)
            path_type = PathType.LINEAR
        else:
            # Has conditionals - extract each branch
            paths = self._extract_conditional_paths_improved(loop_body, variables)
            
            if len(paths) > 2:
                path_type = PathType.MULTI_BRANCH
            elif len(paths) == 2:
                path_type = PathType.IF_THEN_ELSE
            else:
                path_type = PathType.IF_THEN
        
        all_vars = set(variables)
        for path in paths:
            all_vars.update(path.updates.keys())
        
        analysis = LoopPathAnalysis(
            loop_condition=loop_condition,
            paths=paths,
            all_variables=all_vars,
            path_type=path_type,
            has_disjunctive_behavior=len(paths) > 1
        )
        
        return analysis
    
    def _extract_conditional_paths_improved(self, body: str, 
                                           variables: List[str]) -> List[ExecutionPath]:
        """Extract paths from conditional statements with better parsing"""
        paths = []
        
        # Find all if statements with brace matching
        # Pattern: if (...) { ... } [else { ... }]
        i = 0
        while i < len(body):
            # Find 'if'
            if_match = re.search(r'\bif\s*', body[i:])
            if not if_match:
                break
            
            if_start = i + if_match.start()
            after_if = i + if_match.end()
            
            # Find condition - look for either (cond) or just cond before {
            rest = body[after_if:]
            
            # Try to find condition with parentheses
            if rest.lstrip().startswith('('):
                # Find matching paren
                paren_start = rest.find('(')
                paren_count = 0
                cond_end = paren_start
                for j, c in enumerate(rest[paren_start:]):
                    if c == '(':
                        paren_count += 1
                    elif c == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            cond_end = paren_start + j + 1
                            break
                condition = rest[paren_start+1:cond_end-1].strip()
                after_cond = after_if + cond_end
            else:
                # Condition without parens - ends at {
                brace_pos = rest.find('{')
                if brace_pos == -1:
                    break
                condition = rest[:brace_pos].strip()
                after_cond = after_if + brace_pos
            
            # Find then-block by matching braces
            rest = body[after_cond:]
            then_block, then_end = self._extract_brace_block(rest)
            
            if then_block is None:
                break
            
            # Check for else
            after_then = after_cond + then_end
            rest_after_then = body[after_then:].lstrip()
            
            else_block = None
            if rest_after_then.startswith('else'):
                else_start = body.find('else', after_then)
                after_else = else_start + 4
                rest = body[after_else:].lstrip()
                
                if rest.startswith('{'):
                    else_block, _ = self._extract_brace_block(rest)
            
            # Create paths for this if-else
            then_path = self._extract_linear_path(then_block, variables, condition, [])
            paths.append(then_path)
            
            if else_block:
                else_path = self._extract_linear_path(else_block, variables, None, [condition])
                else_path.is_else_branch = True
                paths.append(else_path)
            else:
                # Implicit else path
                implicit_else = ExecutionPath(
                    path_id=self.path_counter + 1,
                    condition=None,
                    negated_conditions=[condition],
                    updates={},
                    is_else_branch=True
                )
                self.path_counter += 1
                paths.append(implicit_else)
            
            # Move past this if-else
            i = after_then
            if else_block:
                i = body.find('}', i) + 1 if '}' in body[i:] else len(body)
        
        # Extract updates outside conditionals
        outer_updates = self._extract_outer_updates_improved(body, variables, paths)
        for path in paths:
            for var, update in outer_updates.items():
                if var not in path.updates:
                    path.updates[var] = update
        
        return paths if paths else [self._extract_linear_path(body, variables, None, [])]
    
    def _extract_brace_block(self, text: str) -> Tuple[Optional[str], int]:
        """Extract content between matching braces"""
        text = text.lstrip()
        if not text.startswith('{'):
            return None, 0
        
        brace_count = 0
        content = ""
        end_pos = 0
        started = False
        
        for i, c in enumerate(text):
            if c == '{':
                brace_count += 1
                if not started:
                    started = True
                    continue
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
            
            if started:
                content += c
        
        return content.strip(), end_pos
    
    def _extract_outer_updates_improved(self, body: str, variables: List[str],
                                        paths: List[ExecutionPath]) -> Dict[str, VariableUpdate]:
        """Extract updates outside conditional blocks"""
        # Simple approach: find assignments that aren't inside any if/else block
        updates = {}
        
        # Look for assignments at the outer level
        lines = body.split('\n')
        in_conditional = 0
        
        for line in lines:
            # Track brace depth (simplified)
            in_conditional += line.count('{') - line.count('}')
            
            # Skip lines inside conditionals
            if 'if' in line or in_conditional > 1:
                continue
            
            for var in variables:
                update = self._find_variable_update(line, var)
                if update and var not in updates:
                    updates[var] = update
        
        return updates
    
    def _extract_linear_path(self, body: str, variables: List[str],
                             condition: Optional[str],
                             negated: List[str]) -> ExecutionPath:
        """Extract updates from a linear (non-branching) code block"""
        updates = {}
        
        for var in variables:
            update = self._find_variable_update(body, var)
            if update:
                updates[var] = update
        
        self.path_counter += 1
        return ExecutionPath(
            path_id=self.path_counter,
            condition=condition,
            negated_conditions=negated,
            updates=updates
        )
    
    def _extract_conditional_paths(self, body: str, variables: List[str],
                                   if_matches: List) -> List[ExecutionPath]:
        """Extract paths from conditional statements"""
        paths = []
        
        for match in if_matches:
            condition = match.group(1).strip()
            then_block = match.group(2)
            else_block = match.group(3) if match.group(3) else None
            
            # Then branch
            then_path = self._extract_linear_path(
                then_block, variables, condition, []
            )
            paths.append(then_path)
            
            # Else branch (if exists)
            if else_block:
                else_path = self._extract_linear_path(
                    else_block, variables, None, [condition]
                )
                else_path.is_else_branch = True
                paths.append(else_path)
            else:
                # Implicit else - no updates for this path
                # But we still need to track it for disjunctive invariants
                implicit_else = ExecutionPath(
                    path_id=self.path_counter + 1,
                    condition=None,
                    negated_conditions=[condition],
                    updates={},
                    is_else_branch=True
                )
                self.path_counter += 1
                paths.append(implicit_else)
        
        # Also extract any updates outside the conditionals
        # (before or after the if statements)
        outer_updates = self._extract_outer_updates(body, variables, if_matches)
        
        # Add outer updates to all paths
        for path in paths:
            for var, update in outer_updates.items():
                if var not in path.updates:
                    path.updates[var] = update
        
        return paths
    
    def _extract_outer_updates(self, body: str, variables: List[str],
                               if_matches: List) -> Dict[str, VariableUpdate]:
        """Extract updates that happen outside conditional blocks"""
        # Remove conditional blocks from body
        outer_body = body
        for match in sorted(if_matches, key=lambda m: m.start(), reverse=True):
            outer_body = outer_body[:match.start()] + outer_body[match.end():]
        
        updates = {}
        for var in variables:
            update = self._find_variable_update(outer_body, var)
            if update:
                updates[var] = update
        
        return updates
    
    def _find_variable_update(self, code: str, var: str) -> Optional[VariableUpdate]:
        """Find how a variable is updated in a code block"""
        # Pattern: var := var + n or var := var - n
        increment_pattern = rf'{var}\s*:=\s*{var}\s*\+\s*(\d+)'
        decrement_pattern = rf'{var}\s*:=\s*{var}\s*-\s*(\d+)'
        multiply_pattern = rf'{var}\s*:=\s*{var}\s*\*\s*(\d+)'
        assign_pattern = rf'{var}\s*:=\s*([^;]+);'
        
        # Check increment
        match = re.search(increment_pattern, code)
        if match:
            delta = int(match.group(1))
            return VariableUpdate(var, f"{var} + {delta}", delta)
        
        # Check decrement
        match = re.search(decrement_pattern, code)
        if match:
            delta = -int(match.group(1))
            return VariableUpdate(var, f"{var} - {abs(delta)}", delta)
        
        # Check multiply
        match = re.search(multiply_pattern, code)
        if match:
            factor = int(match.group(1))
            return VariableUpdate(var, f"{var} * {factor}", None)
        
        # Check general assignment
        match = re.search(assign_pattern, code)
        if match:
            expr = match.group(1).strip()
            return VariableUpdate(var, expr, None)
        
        return None
    
    def get_path_specific_updates(self, analysis: LoopPathAnalysis) -> Dict[int, Dict[str, int]]:
        """
        Get the delta updates for each path.
        Returns: {path_id: {var: delta}}
        """
        result = {}
        for path in analysis.paths:
            deltas = {}
            for var, update in path.updates.items():
                if update.delta is not None:
                    deltas[var] = update.delta
                else:
                    deltas[var] = 0  # Unknown update
            result[path.path_id] = deltas
        return result
    
    def summarize_analysis(self, analysis: LoopPathAnalysis) -> str:
        """Generate a human-readable summary of the path analysis"""
        lines = [
            f"Loop Analysis Summary",
            f"  Condition: {analysis.loop_condition}",
            f"  Path Type: {analysis.path_type.value}",
            f"  Number of Paths: {len(analysis.paths)}",
            f"  Requires Disjunctive Invariant: {analysis.requires_disjunctive_invariant()}",
            f"  Variables: {', '.join(analysis.all_variables)}",
            "",
            "Paths:"
        ]
        
        for path in analysis.paths:
            path_cond = path.get_path_condition()
            lines.append(f"  Path {path.path_id}: {path_cond}")
            for var, update in path.updates.items():
                lines.append(f"    {update}")
        
        return "\n".join(lines)


def analyze_dafny_loop(loop_info: Dict) -> LoopPathAnalysis:
    """
    Convenience function to analyze a loop from parsed Dafny info.
    
    Args:
        loop_info: Dictionary with 'condition', 'body', 'variables' keys
    
    Returns:
        LoopPathAnalysis
    """
    analyzer = PathAnalyzer()
    return analyzer.analyze_loop(
        loop_condition=loop_info.get('condition', 'true'),
        loop_body=loop_info.get('body', ''),
        variables=loop_info.get('variables', [])
    )


if __name__ == "__main__":
    # Test the path analyzer
    analyzer = PathAnalyzer()
    
    # Test 1: Simple linear loop
    print("Test 1: Linear loop")
    analysis = analyzer.analyze_loop(
        "i < n",
        "{ i := i + 1; sum := sum + i; }",
        ["i", "sum", "n"]
    )
    print(analyzer.summarize_analysis(analysis))
    print()
    
    # Test 2: If-then-else loop
    print("Test 2: If-then-else loop")
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
        ["i", "x", "n"]
    )
    print(analyzer.summarize_analysis(analysis))
    print()
    
    # Test 3: If-then (no else) loop
    print("Test 3: If-then (no else) loop")
    analysis = analyzer.analyze_loop(
        "i < n",
        """{ 
            if (x > 0) {
                y := y + 1;
            }
            x := x - 1;
            i := i + 1;
        }""",
        ["i", "x", "y", "n"]
    )
    print(analyzer.summarize_analysis(analysis))
