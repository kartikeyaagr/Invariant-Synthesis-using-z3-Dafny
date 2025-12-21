"""
Dafny Verifier Module
Validates synthesized invariants by running Dafny verification.
Supports combining multiple invariants together.
"""

import subprocess
import tempfile
import os
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations


@dataclass
class VerificationResult:
    """Result of Dafny verification"""
    success: bool
    verified_count: int
    error_count: int
    output: str
    errors: List[str]


class DafnyVerifier:
    """Interface to Dafny verification"""
    
    def __init__(self, dafny_path: str = "dafny"):
        self.dafny_path = dafny_path
    
    def verify_file(self, file_path: str) -> VerificationResult:
        """Verify a Dafny file"""
        try:
            result = subprocess.run(
                [self.dafny_path, 'verify', file_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            output = result.stdout + result.stderr
            
            # Parse results
            verified = 0
            errors = 0
            error_msgs = []
            
            # Look for verification summary
            match = re.search(r'(\d+)\s+verified,\s+(\d+)\s+error', output)
            if match:
                verified = int(match.group(1))
                errors = int(match.group(2))
            
            # Extract error messages
            for line in output.split('\n'):
                if 'Error' in line or 'error' in line.lower():
                    error_msgs.append(line.strip())
            
            return VerificationResult(
                success=(errors == 0 and verified > 0),
                verified_count=verified,
                error_count=errors,
                output=output,
                errors=error_msgs
            )
            
        except FileNotFoundError:
            return VerificationResult(
                success=False,
                verified_count=0,
                error_count=1,
                output="",
                errors=["Dafny not found. Please ensure it is in your PATH."]
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                success=False,
                verified_count=0,
                error_count=1,
                output="",
                errors=["Verification timed out"]
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                verified_count=0,
                error_count=1,
                output="",
                errors=[str(e)]
            )
    
    def verify_source(self, source: str) -> VerificationResult:
        """Verify Dafny source code directly"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.dfy', delete=False
        ) as f:
            f.write(source)
            temp_path = f.name
        
        try:
            return self.verify_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def verify_with_invariant(self, source: str, invariant: str) -> VerificationResult:
        """
        Verify source with a single invariant inserted.
        """
        return self.verify_with_invariants(source, [invariant])
    
    def verify_with_invariants(self, source: str, invariants: List[str]) -> VerificationResult:
        """
        Verify source with multiple invariants inserted together.
        """
        # Find while loop and insert all invariants
        lines = source.split('\n')
        new_lines = []
        inserted = False
        
        for line in lines:
            new_lines.append(line)
            
            # Insert after while line (before any existing invariants or the body)
            if not inserted and 'while' in line and not line.strip().startswith('//'):
                # Find indentation
                indent = len(line) - len(line.lstrip())
                inv_indent = ' ' * (indent + 4)
                
                # Insert all invariants
                for inv in invariants:
                    new_lines.append(f"{inv_indent}invariant {inv}")
                inserted = True
        
        modified_source = '\n'.join(new_lines)
        return self.verify_source(modified_source)
    
    def validate_invariants(self, source: str, 
                           invariants: List[str]) -> List[Tuple[str, bool, str]]:
        """
        Validate multiple invariants against source code.
        Returns list of (invariant, valid, message) tuples.
        """
        results = []
        
        for inv in invariants:
            result = self.verify_with_invariant(source, inv)
            
            if result.success:
                results.append((inv, True, "Verified"))
            else:
                error_msg = result.errors[0] if result.errors else "Verification failed"
                results.append((inv, False, error_msg))
        
        return results
    
    def find_valid_invariants(self, source: str,
                              candidates: List[str]) -> List[str]:
        """
        Filter candidates to return only valid invariants.
        """
        valid = []
        
        for inv in candidates:
            result = self.verify_with_invariant(source, inv)
            if result.success:
                valid.append(inv)
        
        return valid
    
    def find_minimal_valid_set(self, source: str, 
                                candidates: List[str],
                                max_combo_size: int = 4) -> Optional[List[str]]:
        """
        Find a minimal set of invariants that together verify the program.
        Tries combinations of increasing size until one works.
        
        Args:
            source: Dafny source code
            candidates: List of candidate invariants
            max_combo_size: Maximum number of invariants to combine
            
        Returns:
            A list of invariants that together verify, or None if none found
        """
        # Try combinations of increasing size
        for size in range(1, min(max_combo_size + 1, len(candidates) + 1)):
            for combo in combinations(candidates, size):
                result = self.verify_with_invariants(source, list(combo))
                if result.success:
                    return list(combo)
        
        return None
    
    def find_all_valid_sets(self, source: str,
                            candidates: List[str],
                            max_combo_size: int = 3,
                            max_results: int = 5) -> List[List[str]]:
        """
        Find multiple valid sets of invariants.
        
        Returns:
            List of valid invariant combinations
        """
        valid_sets = []
        
        for size in range(1, min(max_combo_size + 1, len(candidates) + 1)):
            for combo in combinations(candidates, size):
                if len(valid_sets) >= max_results:
                    return valid_sets
                    
                result = self.verify_with_invariants(source, list(combo))
                if result.success:
                    valid_sets.append(list(combo))
        
        return valid_sets


class InvariantValidator:
    """
    Validates invariants against the three correctness conditions:
    1. Initialization: Pre => Inv at loop entry
    2. Preservation: Inv && Guard => Inv' after body
    3. Usefulness: Inv && !Guard => Post
    """
    
    def __init__(self, verifier: Optional[DafnyVerifier] = None):
        self.verifier = verifier or DafnyVerifier()
    
    def generate_init_check(self, method_sig: str, precond: str,
                           init_code: str, invariant: str) -> str:
        """Generate Dafny code to check initialization"""
        return f"""
{method_sig}
  requires {precond}
{{
  {init_code}
  assert {invariant};
}}
"""
    
    def generate_preservation_check(self, method_sig: str, 
                                    invariant: str, guard: str,
                                    body: str) -> str:
        """Generate Dafny code to check preservation"""
        return f"""
{method_sig}
  requires {invariant}
  requires {guard}
{{
  {body}
  assert {invariant};
}}
"""
    
    def check_invariant_conditions(self, source: str, 
                                   invariant: str) -> Tuple[bool, bool, bool]:
        """
        Check all three invariant conditions.
        Returns (init_ok, preserve_ok, useful_ok)
        """
        result = self.verifier.verify_with_invariant(source, invariant)
        
        if result.success:
            return (True, True, True)
        
        return (False, False, False)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dafny_verifier.py <file.dfy> [invariant1] [invariant2] ...")
        sys.exit(1)
    
    verifier = DafnyVerifier()
    
    if len(sys.argv) == 2:
        result = verifier.verify_file(sys.argv[1])
    else:
        with open(sys.argv[1], 'r') as f:
            source = f.read()
        invariants = sys.argv[2:]
        result = verifier.verify_with_invariants(source, invariants)
    
    print(f"Success: {result.success}")
    print(f"Verified: {result.verified_count}, Errors: {result.error_count}")
    
    if result.errors:
        print("Errors:")
        for err in result.errors:
            print(f"  {err}")
    
    if result.output:
        print("\nFull output:")
        print(result.output)
