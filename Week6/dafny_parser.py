import re
import json
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class TokenType(Enum):
    METHOD = "method"
    WHILE = "while"
    FOR = "for"
    REQUIRES = "requires"
    ENSURES = "ensures"
    INVARIANT = "invariant"
    VAR = "var"
    IDENTIFIER = "identifier"
    OPERATOR = "operator"
    LITERAL = "literal"
    SEMICOLON = "semicolon"
    LBRACE = "lbrace"
    RBRACE = "rbrace"
    LPAREN = "lparen"
    RPAREN = "rparen"
    EOF = "eof"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int
    pos: int

@dataclass
class LoopInfo:
    variables: List[str]
    loop_type: str
    condition: str
    body: str
    invariants: List[str]

@dataclass
class MethodInfo:
    name: str
    parameters: List[str]
    preconditions: List[str]
    postconditions: List[str]
    loops: List[LoopInfo]
    local_variables: List[str]

class DafnyLexer:
    """Tokenizes Dafny source code"""
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
        # Dafny keywords
        self.keywords = {
            'method': TokenType.METHOD,
            'while': TokenType.WHILE,
            'for': TokenType.FOR,
            'requires': TokenType.REQUIRES,
            'ensures': TokenType.ENSURES,
            'invariant': TokenType.INVARIANT,
            'var': TokenType.VAR,
        }
    
    def current_char(self) -> Optional[str]:
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        peek_pos = self.pos + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def advance(self):
        if self.pos < len(self.source):
            if self.source[self.pos] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.pos += 1
    
    def skip_whitespace(self):
        while (char := self.current_char()) is not None and char.isspace():
            self.advance()
    
    def skip_comment(self):
        if self.current_char() == '/' and self.peek_char() == '/':
            # Single line comment
            while self.current_char() and self.current_char() != '\n':
                self.advance()
        elif self.current_char() == '/' and self.peek_char() == '*':
            # Multi-line comment
            self.advance()  # skip '/'
            self.advance()  # skip '*'
            while self.current_char():
                if self.current_char() == '*' and self.peek_char() == '/':
                    self.advance()  # skip '*'
                    self.advance()  # skip '/'
                    break
                self.advance()
    
    def read_string(self) -> str:
        quote_char = self.current_char()
        self.advance()  # skip opening quote
        value = ''
        while True:
            ch = self.current_char()
            if ch is None or ch == quote_char:
                break
            if ch == '\\':
                self.advance()
                esc = self.current_char()
                if esc is None:
                    break
                value += esc
                self.advance()
            else:
                value += ch
                self.advance()
        if self.current_char() == quote_char:
            self.advance()  # skip closing quote
        return value
    
    def read_number(self) -> str:
        value = ''
        current = self.current_char()
        while current is not None and (current.isdigit() or current == '.'):
            value += current
            self.advance()
            current = self.current_char()
        return value
    
    def read_identifier(self) -> str:
        value = ''
        char = self.current_char()
        while char is not None and (char.isalnum() or char == '_'):
            value += char
            self.advance()
            char = self.current_char()
        return value
    
    def tokenize(self) -> List[Token]:
        while self.current_char():
            self.skip_whitespace()
            
            if not self.current_char():
                break
                
            line, column, pos = self.line, self.column, self.pos
            
            # Comments
            if self.current_char() == '/' and self.peek_char() in ['/', '*']:
                self.skip_comment()
                continue
            
            # Strings
            if self.current_char() in ['"', "'"]:
                value = self.read_string()
                self.tokens.append(Token(TokenType.LITERAL, value, line, column, pos))
                continue
            
            # Numbers
            char = self.current_char()
            if char is not None and char.isdigit():
                value = self.read_number()
                self.tokens.append(Token(TokenType.LITERAL, value, line, column, pos))
                continue
            
            # Identifiers and keywords
            current = self.current_char()
            if current is not None and (current.isalpha() or current == '_'):
                value = self.read_identifier()
                token_type = self.keywords.get(value, TokenType.IDENTIFIER)
                self.tokens.append(Token(token_type, value, line, column, pos))
                continue

            # Operators and other symbols
            char = self.current_char()
            peek = self.peek_char()
            
            # Multi-character operators first
            if char == ':' and peek == '=':
                self.tokens.append(Token(TokenType.OPERATOR, ':=', line, column, pos))
                self.advance()
                self.advance()
                continue
            if char == '>' and peek == '=':
                self.tokens.append(Token(TokenType.OPERATOR, '>=', line, column, pos))
                self.advance()
                self.advance()
                continue
            if char == '<' and peek == '=':
                self.tokens.append(Token(TokenType.OPERATOR, '<=', line, column, pos))
                self.advance()
                self.advance()
                continue
            if char == '=' and peek == '=':
                self.tokens.append(Token(TokenType.OPERATOR, '==', line, column, pos))
                self.advance()
                self.advance()
                continue
            if char == '!' and peek == '=':
                self.tokens.append(Token(TokenType.OPERATOR, '!=', line, column, pos))
                self.advance()
                self.advance()
                continue

            # Single character tokens
            if char == ';':
                self.tokens.append(Token(TokenType.SEMICOLON, char, line, column, pos))
                self.advance()
                continue
            if char == '{':
                self.tokens.append(Token(TokenType.LBRACE, char, line, column, pos))
                self.advance()
                continue
            if char == '}':
                self.tokens.append(Token(TokenType.RBRACE, char, line, column, pos))
                self.advance()
                continue
            if char == '(':
                self.tokens.append(Token(TokenType.LPAREN, char, line, column, pos))
                self.advance()
                continue
            if char == ')':
                self.tokens.append(Token(TokenType.RPAREN, char, line, column, pos))
                self.advance()
                continue
            
            # Assume anything else is a single character operator
            if char is not None:
                self.tokens.append(Token(TokenType.OPERATOR, char, line, column, pos))
                self.advance()
                continue

        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column, self.pos))
        return self.tokens

class DafnyParser:
    """Parses tokenized Dafny code and extracts relevant information"""
    
    def __init__(self, tokens: List[Token], source: str):
        self.tokens = tokens
        self.pos = 0
        self.methods = []
        self.source = source
    
    def current_token(self) -> Token:
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # EOF token
        return self.tokens[self.pos]
    
    def peek_token(self, offset: int = 1) -> Token:
        peek_pos = self.pos + offset
        if peek_pos >= len(self.tokens):
            return self.tokens[-1]  # EOF token
        return self.tokens[peek_pos]
    
    def advance(self):
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
    
    def expect(self, token_type: TokenType) -> Token:
        token = self.current_token()
        if token.type != token_type:
            raise SyntaxError(f"Expected {token_type}, got {token.type} at line {token.line}")
        self.advance()
        return token
    
    def parse_method_signature(self) -> tuple[str, List[str]]:
        """Parse method name and parameters"""
        method_name = self.expect(TokenType.IDENTIFIER).value
        
        # Skip to parameters
        self.expect(TokenType.LPAREN)
        
        parameters = []
        while self.current_token().type != TokenType.RPAREN:
            if self.current_token().type == TokenType.IDENTIFIER:
                param_name = self.current_token().value
                parameters.append(param_name)
            self.advance()
        
        self.expect(TokenType.RPAREN)
        
        # Skip return type if present
        while self.current_token().type not in [TokenType.REQUIRES, TokenType.ENSURES, TokenType.LBRACE, TokenType.EOF]:
            self.advance()
        
        return method_name, parameters
    
    def parse_condition(self) -> str:
        """Parse a condition (for requires, ensures, while, etc.)"""
        tokens = []
        while self.current_token().type not in [
            TokenType.EOF, TokenType.LBRACE, TokenType.RBRACE,
            TokenType.REQUIRES, TokenType.ENSURES, TokenType.INVARIANT,
            TokenType.WHILE, TokenType.FOR, TokenType.METHOD, TokenType.SEMICOLON
        ]:
            tokens.append(self.current_token().value)
            self.advance()
        return ' '.join(tokens).strip()

    def parse_preconditions(self) -> List[str]:
        """Parse all requires clauses"""
        preconditions = []
        while self.current_token().type == TokenType.REQUIRES:
            self.advance()  # skip 'requires'
            condition = self.parse_condition()
            preconditions.append(condition)
        return preconditions
    
    def parse_postconditions(self) -> List[str]:
        """Parse all ensures clauses"""
        postconditions = []
        while self.current_token().type == TokenType.ENSURES:
            self.advance()  # skip 'ensures'
            condition = self.parse_condition()
            postconditions.append(condition)
        return postconditions
    
    def extract_variables_from_expression(self, expr: str) -> List[str]:
        """Extract variable names from an expression"""
        # Simple regex to find identifiers (variables)
        identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        identifiers = re.findall(identifier_pattern, expr)
        
        # Filter out keywords and operators
        keywords = {'int', 'bool', 'real', 'true', 'false', 'old', 'forall', 'exists'}
        variables = [id for id in identifiers if id not in keywords]
        
        return list(set(variables))  # Remove duplicates
    
    def parse_while_loop(self) -> LoopInfo:
        """Parse a while loop"""
        self.advance()  # skip 'while'
        
        # Parse condition
        self.expect(TokenType.LPAREN)
        condition_tokens = []
        paren_count = 1
        
        while paren_count > 0 and self.current_token().type != TokenType.EOF:
            token = self.current_token()
            if token.type == TokenType.LPAREN:
                paren_count += 1
            elif token.type == TokenType.RPAREN:
                paren_count -= 1
            
            if paren_count > 0:  # Don't include the closing paren
                condition_tokens.append(token.value)
            self.advance()
        
        condition = ' '.join(condition_tokens).strip()
        
        # Extract variables from condition
        loop_variables = self.extract_variables_from_expression(condition)
        
        # Parse invariants
        invariants = []
        while self.current_token().type == TokenType.INVARIANT:
            self.advance()  # skip 'invariant'
            invariant_condition = self.parse_condition()
            invariants.append(invariant_condition)
        
        # Parse loop body
        self.expect(TokenType.LBRACE)
        body_start_pos = self.tokens[self.pos - 1].pos
        
        brace_count = 1
        while brace_count > 0 and self.current_token().type != TokenType.EOF:
            token = self.current_token()
            if token.type == TokenType.LBRACE:
                brace_count += 1
            elif token.type == TokenType.RBRACE:
                brace_count -= 1
            self.advance()
        
        body_end_pos = self.tokens[self.pos - 1].pos
        loop_body = self.source[body_start_pos:body_end_pos]

        return LoopInfo(
            variables=loop_variables,
            loop_type="while",
            condition=condition,
            body=loop_body,
            invariants=invariants
        )
    
    def parse_for_loop(self) -> LoopInfo:
        """Parse a for loop (simplified implementation)"""
        self.advance()  # skip 'for'
        
        condition_tokens = []
        while self.current_token().type not in [TokenType.INVARIANT, TokenType.LBRACE, TokenType.EOF]:
            condition_tokens.append(self.current_token().value)
            self.advance()
        
        condition = ' '.join(condition_tokens).strip()
        loop_variables = self.extract_variables_from_expression(condition)

        # Parse invariants
        invariants = []
        while self.current_token().type == TokenType.INVARIANT:
            self.advance()  # skip 'invariant'
            invariant_condition = self.parse_condition()
            invariants.append(invariant_condition)

        # Parse loop body
        self.expect(TokenType.LBRACE)
        body_start_pos = self.tokens[self.pos - 1].pos

        brace_count = 1
        while brace_count > 0 and self.current_token().type != TokenType.EOF:
            token = self.current_token()
            if token.type == TokenType.LBRACE:
                brace_count += 1
            elif token.type == TokenType.RBRACE:
                brace_count -= 1
            self.advance()
        
        body_end_pos = self.tokens[self.pos - 1].pos
        loop_body = self.source[body_start_pos:body_end_pos]

        return LoopInfo(
            variables=loop_variables,
            loop_type="for",
            condition=condition,
            body=loop_body,
            invariants=invariants
        )
    
    def parse_method_body(self) -> tuple[List[LoopInfo], List[str]]:
        """Parse method body and extract loops and local variables"""
        loops = []
        local_variables = []
        
        self.expect(TokenType.LBRACE)
        
        brace_count = 1
        while brace_count > 0 and self.current_token().type != TokenType.EOF:
            token = self.current_token()
            
            if token.type == TokenType.LBRACE:
                brace_count += 1
                self.advance()
            elif token.type == TokenType.RBRACE:
                brace_count -= 1
                self.advance()
            elif token.type == TokenType.WHILE:
                loop_info = self.parse_while_loop()
                loops.append(loop_info)
            elif token.type == TokenType.FOR:
                loop_info = self.parse_for_loop()
                loops.append(loop_info)
            elif token.type == TokenType.VAR:
                self.advance()  # skip 'var'
                if self.current_token().type == TokenType.IDENTIFIER:
                    var_name = self.current_token().value
                    local_variables.append(var_name)
                self.advance()
            else:
                self.advance()
        
        return loops, local_variables
    
    def parse_method(self) -> MethodInfo:
        """Parse a complete method"""
        self.expect(TokenType.METHOD)
        
        # Parse method signature
        method_name, parameters = self.parse_method_signature()
        
        # Parse pre-conditions
        preconditions = self.parse_preconditions()
        
        # Parse post-conditions
        postconditions = self.parse_postconditions()
        
        # Parse method body
        loops, local_variables = self.parse_method_body()
        
        return MethodInfo(
            name=method_name,
            parameters=parameters,
            preconditions=preconditions,
            postconditions=postconditions,
            loops=loops,
            local_variables=local_variables
        )
    
    def parse(self) -> List[MethodInfo]:
        """Parse the entire program"""
        methods = []
        
        while self.current_token().type != TokenType.EOF:
            if self.current_token().type == TokenType.METHOD:
                method_info = self.parse_method()
                methods.append(method_info)
            else:
                self.advance()
        
        return methods

class DafnyExtractor:
    """Main class for extracting information from Dafny programs"""
    
    def __init__(self):
        pass
    
    def parse_file(self, filename: str) -> Dict[str, Any]:
        """Parse a Dafny file and extract information"""
        try:
            with open(filename, 'r') as f:
                source = f.read()
            return self.parse_source(source)
        except FileNotFoundError:
            return {"error": f"File not found: {filename}"}
        except Exception as e:
            return {"error": f"Error parsing file: {str(e)}"}
    
    def parse_source(self, source: str) -> Dict[str, Any]:
        """Parse Dafny source code and extract information"""
        try:
            # Tokenize
            lexer = DafnyLexer(source)
            tokens = lexer.tokenize()
            
            # Parse
            parser = DafnyParser(tokens, source)
            methods = parser.parse()
            
            # Extract information in the required format
            result = {
                "methods": [],
                "loops": [],
                "preconditions": [],
                "postconditions": []
            }
            
            for method in methods:
                method_data = {
                    "name": method.name,
                    "parameters": method.parameters,
                    "preconditions": method.preconditions,
                    "postconditions": method.postconditions,
                    "local_variables": method.local_variables,
                    "loops": []
                }
                
                for loop in method.loops:
                    loop_data = {
                        "variables": loop.variables,
                        "loop_type": loop.loop_type,
                        "condition": loop.condition,
                        "body": loop.body,
                        "invariants": loop.invariants
                    }
                    method_data["loops"].append(loop_data)
                    result["loops"].append(loop_data)
                
                result["methods"].append(method_data)
                result["preconditions"].extend(method.preconditions)
                result["postconditions"].extend(method.postconditions)
            
            return result
            
        except Exception as e:
            return {"error": f"Error parsing source: {str(e)}"}

def main():
    """Main function for command-line usage"""
    if len(sys.argv) != 2:
        print("Usage: python dafny_parser.py <dafny_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    extractor = DafnyExtractor()
    result = extractor.parse_file(filename)
    
    output_filename = "".join(filename.split('.')[:-1]) + ".json"
    with open(output_filename, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Output written to {output_filename}")

if __name__ == "__main__":
    main()