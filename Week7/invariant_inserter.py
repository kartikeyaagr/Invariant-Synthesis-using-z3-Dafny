
import json

def insert_invariants(lines, loop_info, invariants):
    new_lines = lines[:]
    indentation = ' ' * (loop_info['indentation'] + 2)
    
    for inv in reversed(invariants['invariants']):
        new_lines.insert(loop_info['line_number'] + 1, f"{indentation}invariant {inv}\n")
        
    return "".join(new_lines)

if __name__ == '__main__':
    from program_parser import parse_dafny_program

    file_path = 'test_program.dfy'
    lines, loop_info = parse_dafny_program(file_path)
    
    with open('invariants.json', 'r') as f:
        invariants = json.load(f)
        
    new_program = insert_invariants(lines, loop_info, invariants)
    
    with open('test_program_with_invariants.dfy', 'w') as f:
        f.write(new_program)
        
    print("Generated Dafny program with invariants:")
    print(new_program)
