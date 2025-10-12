
import json
from program_parser import parse_dafny_program
from invariant_inserter import insert_invariants
from dafny_verifier import verify_dafny_program

def main():
    original_program_path = 'test_program.dfy'
    invariants_path = 'invariants.json'
    output_program_path = 'test_program_with_invariants.dfy'

    # Parse og program
    print(f"Parsing program: {original_program_path}")
    lines, loop_info = parse_dafny_program(original_program_path)
    print(f"Found loop at line {loop_info['line_number']}")

    # load invariants from json
    print(f"Loading invariants from: {invariants_path}")
    with open(invariants_path, 'r') as f:
        invariants = json.load(f)
    print(f"Loaded {len(invariants['invariants'])} invariants.")

    # insert invariants into program
    print("Inserting invariants into the program...")
    new_program_content = insert_invariants(lines, loop_info, invariants)

    # Write new file with invariants
    print(f"Writing new program to: {output_program_path}")
    with open(output_program_path, 'w') as f:
        f.write(new_program_content)
    print("New program written successfully.")

    # Verify using dafny
    print(f"Running Dafny verifier on: {output_program_path}")
    verification_result = verify_dafny_program(output_program_path)

    # Result
    print("\n--- Verification Result ---")
    print(verification_result)

if __name__ == "__main__":
    main()
