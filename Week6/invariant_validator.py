import subprocess
import tempfile

class InvariantValidator:
    """Validates an invariant by instrumenting a Dafny program and running Dafny."""

    def validate(self, dafny_file_path: str, invariant: str) -> bool:
        """Validates the given invariant for the given Dafny program.

        Args:
            dafny_file_path: The path to the Dafny program.
            invariant: The invariant to validate.

        Returns:
            True if the invariant is valid, False otherwise.
        """
        try:
            with open(dafny_file_path, 'r') as f:
                dafny_code = f.read()

            # Find the while loop and insert the invariant.
            # This is a simplistic approach and might not work for all programs.
            # A more robust solution would use the parser to identify the loop and insert the invariant.
            loop_keyword = 'while'
            loop_index = dafny_code.find(loop_keyword)
            if loop_index == -1:
                return False # No while loop found

            # Find the opening brace of the loop body
            brace_index = dafny_code.find('{', loop_index)
            if brace_index == -1:
                return False # No loop body found

            # Insert the invariant
            instrumented_code = (
                dafny_code[:brace_index] +
                f'\n    invariant {invariant}\n' +
                dafny_code[brace_index:]
            )

            # Write the instrumented code to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.dfy', delete=False) as temp_file:
                temp_file.write(instrumented_code)
                temp_file_path = temp_file.name

            # Run Dafny on the temporary file
            result = subprocess.run(['dafny', 'verify', temp_file_path], capture_output=True, text=True)

            # Check the output for verification results
            return 'Dafny program verifier finished with 1 verified, 0 errors' in result.stdout

        except FileNotFoundError:
            print("Dafny not found. Please make sure it is in your PATH.")
            return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
