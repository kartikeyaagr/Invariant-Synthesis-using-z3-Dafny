"""
Array Property Analyzer for Invariant Synthesis
Analyzes array operations and identifies common patterns for invariant generation.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class ArrayOperation(Enum):
    """Types of array operations"""
    READ = "read"           # a[i]
    WRITE = "write"         # a[i] := x
    LENGTH = "length"       # a.Length
    CREATE = "create"       # new int[n]
    SLICE = "slice"         # a[i..j]


class ArrayPattern(Enum):
    """Common array loop patterns"""
    LINEAR_SCAN = "linear_scan"           # i goes 0 to n
    REVERSE_SCAN = "reverse_scan"         # i goes n-1 to 0
    TWO_POINTER = "two_pointer"           # two indices moving
    PARTITION = "partition"               # elements split by condition
    ACCUMULATE = "accumulate"             # sum/product over array
    SEARCH = "search"                     # find element
    TRANSFORM = "transform"               # map operation
    FILTER = "filter"                     # select elements
    COPY = "copy"                         # copy array
    INIT = "init"                         # initialize array


@dataclass
class ArrayInfo:
    """Information about an array variable"""
    name: str
    element_type: str = "int"
    is_parameter: bool = False
    is_mutable: bool = True
    length_var: Optional[str] = None  # Variable tracking length


@dataclass
class ArrayAccess:
    """Represents an array access operation"""
    array: str
    index: str
    operation: ArrayOperation
    value: Optional[str] = None  # For writes


@dataclass
class LoopArrayAnalysis:
    """Analysis of array operations within a loop"""
    arrays: Dict[str, ArrayInfo]
    accesses: List[ArrayAccess]
    index_var: str
    index_init: int
    index_bound: str
    index_direction: str  # "increasing" or "decreasing"
    pattern: ArrayPattern
    modified_arrays: Set[str]
    read_arrays: Set[str]


class ArrayAnalyzer:
    """Analyzes array operations in loop bodies"""
    
    def __init__(self):
        self.array_pattern = re.compile(r'(\w+)\s*\[\s*([^\]]+)\s*\]')
        self.length_pattern = re.compile(r'(\w+)\.Length')
        self.write_pattern = re.compile(r'(\w+)\s*\[\s*([^\]]+)\s*\]\s*:=\s*([^;]+)')
        self.new_array_pattern = re.compile(r'new\s+(\w+)\s*\[\s*([^\]]+)\s*\]')
    
    def analyze_loop(self, loop_condition: str, loop_body: str,
                     variables: List[str], arrays: List[str] = None) -> LoopArrayAnalysis:
        """
        Analyze array operations in a loop.
        
        Args:
            loop_condition: The while loop guard
            loop_body: Body of the loop
            variables: All variables in scope
            arrays: Known array variables (auto-detected if None)
        """
        # Detect arrays if not provided
        if arrays is None:
            arrays = self._detect_arrays(loop_body, variables)
        
        # Build array info
        array_infos = {name: ArrayInfo(name=name) for name in arrays}
        
        # Find array accesses
        accesses = self._extract_accesses(loop_body, arrays)
        
        # Determine index variable and direction
        index_var, index_init, index_bound, direction = self._analyze_index(
            loop_condition, loop_body, variables
        )
        
        # Identify pattern
        pattern = self._identify_pattern(accesses, index_var, loop_body)
        
        # Track modified vs read arrays
        modified = set()
        read = set()
        for access in accesses:
            if access.operation == ArrayOperation.WRITE:
                modified.add(access.array)
            elif access.operation == ArrayOperation.READ:
                read.add(access.array)
        
        return LoopArrayAnalysis(
            arrays=array_infos,
            accesses=accesses,
            index_var=index_var,
            index_init=index_init,
            index_bound=index_bound,
            index_direction=direction,
            pattern=pattern,
            modified_arrays=modified,
            read_arrays=read
        )
    
    def _detect_arrays(self, code: str, variables: List[str]) -> List[str]:
        """Detect array variables from code patterns"""
        arrays = set()
        
        # Find variables used with [] access
        for match in self.array_pattern.finditer(code):
            arrays.add(match.group(1))
        
        # Find variables with .Length
        for match in self.length_pattern.finditer(code):
            arrays.add(match.group(1))
        
        return list(arrays)
    
    def _extract_accesses(self, code: str, arrays: List[str]) -> List[ArrayAccess]:
        """Extract all array accesses from code"""
        accesses = []
        
        # Find writes first
        for match in self.write_pattern.finditer(code):
            arr, idx, val = match.groups()
            if arr in arrays:
                accesses.append(ArrayAccess(
                    array=arr,
                    index=idx.strip(),
                    operation=ArrayOperation.WRITE,
                    value=val.strip()
                ))
        
        # Find all reads (excluding those that are part of writes)
        write_positions = set()
        for match in self.write_pattern.finditer(code):
            write_positions.add(match.start())
        
        for match in self.array_pattern.finditer(code):
            if match.start() not in write_positions:
                arr, idx = match.groups()
                if arr in arrays:
                    accesses.append(ArrayAccess(
                        array=arr,
                        index=idx.strip(),
                        operation=ArrayOperation.READ
                    ))
        
        # Find length accesses
        for match in self.length_pattern.finditer(code):
            arr = match.group(1)
            if arr in arrays:
                accesses.append(ArrayAccess(
                    array=arr,
                    index="",
                    operation=ArrayOperation.LENGTH
                ))
        
        return accesses
    
    def _analyze_index(self, condition: str, body: str, 
                       variables: List[str]) -> Tuple[str, int, str, str]:
        """Analyze the loop index variable"""
        # Common patterns: i < n, i < a.Length, i >= 0, etc.
        
        # Try to find index from condition
        index_patterns = [
            (r'(\w+)\s*<\s*(\w+(?:\.Length)?)', "increasing"),
            (r'(\w+)\s*<=\s*(\w+(?:\.Length)?)', "increasing"),
            (r'(\w+)\s*>\s*(\d+)', "decreasing"),
            (r'(\w+)\s*>=\s*(\d+)', "decreasing"),
        ]
        
        index_var = "i"
        index_bound = "n"
        direction = "increasing"
        
        for pattern, dir_type in index_patterns:
            match = re.search(pattern, condition)
            if match:
                index_var = match.group(1)
                index_bound = match.group(2)
                direction = dir_type
                break
        
        # Try to find initialization
        init_match = re.search(rf'{index_var}\s*:=\s*(\d+)', body)
        index_init = int(init_match.group(1)) if init_match else 0
        
        return index_var, index_init, index_bound, direction
    
    def _identify_pattern(self, accesses: List[ArrayAccess], 
                         index_var: str, body: str) -> ArrayPattern:
        """Identify the array access pattern"""
        has_writes = any(a.operation == ArrayOperation.WRITE for a in accesses)
        has_reads = any(a.operation == ArrayOperation.READ for a in accesses)
        
        # Check for accumulator pattern (sum, product)
        if re.search(r'\w+\s*:=\s*\w+\s*[+*]\s*\w+\s*\[', body):
            return ArrayPattern.ACCUMULATE
        
        # Check for initialization pattern
        if has_writes and not has_reads:
            return ArrayPattern.INIT
        
        # Check for copy pattern
        if has_writes and has_reads:
            write_arrays = set(a.array for a in accesses if a.operation == ArrayOperation.WRITE)
            read_arrays = set(a.array for a in accesses if a.operation == ArrayOperation.READ)
            if write_arrays != read_arrays:
                return ArrayPattern.COPY
            return ArrayPattern.TRANSFORM
        
        # Default to linear scan
        if has_reads:
            return ArrayPattern.LINEAR_SCAN
        
        return ArrayPattern.LINEAR_SCAN


@dataclass
class QuantifiedInvariant:
    """Represents a quantified invariant"""
    quantifier: str  # "forall" or "exists"
    bound_var: str
    lower_bound: str
    upper_bound: str
    property: str
    
    def to_string(self) -> str:
        return f"{self.quantifier} {self.bound_var} :: {self.lower_bound} <= {self.bound_var} < {self.upper_bound} ==> {self.property}"


class ArrayInvariantTemplates:
    """Templates for common array invariants"""
    
    @staticmethod
    def bounds_invariant(index: str, lower: str, upper: str) -> str:
        """Basic bounds: lower <= index <= upper"""
        return f"{lower} <= {index} <= {upper}"
    
    @staticmethod
    def array_bounds(index: str, array: str) -> str:
        """Array index bounds: 0 <= i <= a.Length"""
        return f"0 <= {index} <= {array}.Length"
    
    @staticmethod
    def processed_elements(index: str, array: str, property: str, 
                          bound_var: str = "k") -> QuantifiedInvariant:
        """
        All processed elements satisfy property:
        forall k :: 0 <= k < i ==> property(a[k])
        """
        return QuantifiedInvariant(
            quantifier="forall",
            bound_var=bound_var,
            lower_bound="0",
            upper_bound=index,
            property=property
        )
    
    @staticmethod
    def accumulator_invariant(acc_var: str, array: str, index: str,
                             operation: str = "+") -> str:
        """
        Accumulator equals operation over processed elements
        """
        if operation == "+":
            return f"{acc_var} == Sum({array}[..{index}])"
        elif operation == "*":
            return f"{acc_var} == Product({array}[..{index}])"
        return f"{acc_var} >= 0"
    
    @staticmethod
    def initialized_elements(index: str, array: str, value: str,
                            bound_var: str = "k") -> QuantifiedInvariant:
        """
        All initialized elements have value:
        forall k :: 0 <= k < i ==> a[k] == value
        """
        return QuantifiedInvariant(
            quantifier="forall",
            bound_var=bound_var,
            lower_bound="0",
            upper_bound=index,
            property=f"{array}[{bound_var}] == {value}"
        )
    
    @staticmethod
    def sorted_prefix(index: str, array: str, bound_var: str = "k") -> QuantifiedInvariant:
        """
        Processed prefix is sorted:
        forall k :: 0 <= k < i-1 ==> a[k] <= a[k+1]
        """
        return QuantifiedInvariant(
            quantifier="forall",
            bound_var=bound_var,
            lower_bound="0",
            upper_bound=f"{index} - 1",
            property=f"{array}[{bound_var}] <= {array}[{bound_var} + 1]"
        )
    
    @staticmethod
    def elements_in_range(index: str, array: str, lower: str, upper: str,
                         bound_var: str = "k") -> QuantifiedInvariant:
        """
        All elements in range:
        forall k :: 0 <= k < i ==> lower <= a[k] <= upper
        """
        return QuantifiedInvariant(
            quantifier="forall",
            bound_var=bound_var,
            lower_bound="0",
            upper_bound=index,
            property=f"{lower} <= {array}[{bound_var}] <= {upper}"
        )
    
    @staticmethod
    def copy_invariant(src: str, dst: str, index: str,
                      bound_var: str = "k") -> QuantifiedInvariant:
        """
        Copy invariant:
        forall k :: 0 <= k < i ==> dst[k] == src[k]
        """
        return QuantifiedInvariant(
            quantifier="forall",
            bound_var=bound_var,
            lower_bound="0",
            upper_bound=index,
            property=f"{dst}[{bound_var}] == {src}[{bound_var}]"
        )
    
    @staticmethod
    def unchanged_suffix(index: str, array: str, old_array: str,
                        bound_var: str = "k") -> QuantifiedInvariant:
        """
        Unprocessed suffix unchanged:
        forall k :: i <= k < a.Length ==> a[k] == old(a[k])
        """
        return QuantifiedInvariant(
            quantifier="forall",
            bound_var=bound_var,
            lower_bound=index,
            upper_bound=f"{array}.Length",
            property=f"{array}[{bound_var}] == {old_array}[{bound_var}]"
        )


if __name__ == "__main__":
    print("Testing Array Analyzer...")
    print("=" * 60)
    
    analyzer = ArrayAnalyzer()
    
    # Test 1: Simple array initialization
    print("\nTest 1: Array initialization")
    analysis = analyzer.analyze_loop(
        "i < a.Length",
        "{ a[i] := 0; i := i + 1; }",
        ["i", "a"],
        ["a"]
    )
    print(f"  Pattern: {analysis.pattern.value}")
    print(f"  Index: {analysis.index_var}")
    print(f"  Modified arrays: {analysis.modified_arrays}")
    
    # Test 2: Array sum
    print("\nTest 2: Array sum")
    analysis = analyzer.analyze_loop(
        "i < a.Length",
        "{ sum := sum + a[i]; i := i + 1; }",
        ["i", "sum", "a"],
        ["a"]
    )
    print(f"  Pattern: {analysis.pattern.value}")
    print(f"  Read arrays: {analysis.read_arrays}")
    
    # Test 3: Array copy
    print("\nTest 3: Array copy")
    analysis = analyzer.analyze_loop(
        "i < src.Length",
        "{ dst[i] := src[i]; i := i + 1; }",
        ["i", "src", "dst"],
        ["src", "dst"]
    )
    print(f"  Pattern: {analysis.pattern.value}")
    print(f"  Read: {analysis.read_arrays}")
    print(f"  Modified: {analysis.modified_arrays}")
    
    # Test templates
    print("\nTest 4: Invariant templates")
    templates = ArrayInvariantTemplates()
    
    print(f"  Bounds: {templates.array_bounds('i', 'a')}")
    
    init_inv = templates.initialized_elements("i", "a", "0")
    print(f"  Initialized: {init_inv.to_string()}")
    
    copy_inv = templates.copy_invariant("src", "dst", "i")
    print(f"  Copy: {copy_inv.to_string()}")
    
    print("\n" + "=" * 60)
    print("Tests completed.")
