import subprocess

def verify_dafny_program(file_path):
    try:
        result = subprocess.run(['dafny', 'verify', file_path], capture_output=True, text=True, check=True)
        return result.stdout
    except FileNotFoundError:
        return "Error: 'dafny' command not found. Please ensure Dafny is installed and in your PATH."
    except subprocess.CalledProcessError as e:
        return e.stdout + e.stderr

if __name__ == '__main__':
    file_path = 'test_program_with_invariants.dfy'
    verification_result = verify_dafny_program(file_path)
    
    print("Dafny verification result:")
    print(verification_result)
