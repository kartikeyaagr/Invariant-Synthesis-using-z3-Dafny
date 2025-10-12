def parse_dafny_program(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    loop_info = {}
    for i, line in enumerate(lines):
        if "while" in line:
            loop_info['line_number'] = i
            loop_info['indentation'] = len(line) - len(line.lstrip())
            break
            
    return lines, loop_info

if __name__ == '__main__':
    file_path = 'test_program.dfy'
    lines, loop_info = parse_dafny_program(file_path)
    
    print(f"Loop found at line: {loop_info['line_number']}")
    print(f"Indentation: {loop_info['indentation']}")
