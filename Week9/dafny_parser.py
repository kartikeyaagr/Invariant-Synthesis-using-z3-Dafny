"""
Dafny Parser for Invariant Synthesis
Extracts loop information, pre/postconditions, and program structure.
"""

import re
import json
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class TokenType(Enum):
    METHOD = "method"
    WHILE = "while"
    FOR = "for"
    IF = "if"
    ELSE = "else"
    REQUIRES = "requires"
    ENSURES = "ensures"
    INVARIANT = "invariant"
    DECREASES = "decreases"
    VAR = "var"
    RETURNS = "returns"
    IDENTIFIER = "identifier"
    OPERATOR = "operator"
    LITERAL = "literal"
    SEMICOLON = "semicolon"
    LBRACE = "lbrace"
    RBRACE = "rbrace"
    LPAREN = "lparen"
    RPAREN = "rparen"
    COLON = "colon"
    COMMA = "comma"
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
    has_conditionals: bool = False
    body_updates: Dict[str, str] = field(default_factory=dict)


@dataclass
class MethodInfo:
    name: str
    parameters: List[tuple]
    return_vars: List[tuple]
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

        self.keywords = {
            'method': TokenType.METHOD,
            'while': TokenType.WHILE,
            'for': TokenType.FOR,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'requires': TokenType.REQUIRES,
            'ensures': TokenType.ENSURES,
            'invariant': TokenType.INVARIANT,
            'decreases': TokenType.DECREASES,
            'var': TokenType.VAR,
            'returns': TokenType.RETURNS,
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
            while self.current_char() and self.current_char() != '\n':
                self.advance()
        elif self.current_char() == '/' and self.peek_char() == '*':
            self.advance()
            self.advance()
            while self.current_char():
                if self.current_char() == '*' and self.peek_char() == '/':
                    self.advance()
                    self.advance()
                    break
                self.advance()

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

            if self.current_char() == '/' and self.peek_char() in ['/', '*']:
                self.skip_comment()
                continue

            char = self.current_char()
            if char is not None and char.isdigit():
                value = self.read_number()
                self.tokens.append(Token(TokenType.LITERAL, value, line, column, pos))
                continue

            current = self.current_char()
            if current is not None and (current.isalpha() or current == '_'):
                value = self.read_identifier()
                token_type = self.keywords.get(value, TokenType.IDENTIFIER)
                self.tokens.append(Token(token_type, value, line, column, pos))
                continue

            char = self.current_char()
            peek = self.peek_char()

            # Multi-character operators
            two_char_ops = [':=', '>=', '<=', '==', '!=', '&&', '||', '==>']
            for op in two_char_ops:
                if char == op[0] and peek == op[1]:
                    self.tokens.append(Token(TokenType.OPERATOR, op, line, column, pos))
                    self.advance()
                    self.advance()
                    break
            else:
                # Single character tokens
                if char == ';':
                    self.tokens.append(Token(TokenType.SEMICOLON, char, line, column, pos))
                elif char == '{':
                    self.tokens.append(Token(TokenType.LBRACE, char, line, column, pos))
                elif char == '}':
                    self.tokens.append(Token(TokenType.RBRACE, char, line, column, pos))
                elif char == '(':
                    self.tokens.append(Token(TokenType.LPAREN, char, line, column, pos))
                elif char == ')':
                    self.tokens.append(Token(TokenType.RPAREN, char, line, column, pos))
                elif char == ':':
                    self.tokens.append(Token(TokenType.COLON, char, line, column, pos))
                elif char == ',':
                    self.tokens.append(Token(TokenType.COMMA, char, line, column, pos))
                elif char in '+-*/%<>=!':
                    self.tokens.append(Token(TokenType.OPERATOR, char, line, column, pos))
                self.advance()

        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column, self.pos))
        return self.tokens


class DafnyParser:
    """Parses Dafny tokens into structured program information"""

    def __init__(self, tokens: List[Token], source: str):
        self.tokens = tokens
        self.source = source
        self.pos = 0

    def current_token(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF, '', 0, 0, len(self.source))

    def advance(self):
        self.pos += 1

    def expect(self, token_type: TokenType):
        if self.current_token().type != token_type:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token().type}")
        self.advance()

    def parse_condition(self) -> str:
        """Parse a condition until we hit a keyword or brace"""
        tokens = []
        stop_types = {TokenType.INVARIANT, TokenType.DECREASES, TokenType.LBRACE, 
                      TokenType.REQUIRES, TokenType.ENSURES, TokenType.EOF}
        
        while self.current_token().type not in stop_types:
            tokens.append(self.current_token().value)
            self.advance()
        return ' '.join(tokens).strip()

    def extract_variables(self, expr: str) -> List[str]:
        """Extract variable names from an expression"""
        identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr)
        keywords = {'while', 'if', 'else', 'var', 'int', 'bool', 'true', 'false', 
                   'requires', 'ensures', 'invariant', 'method', 'returns', 'to'}
        return list(set(v for v in identifiers if v not in keywords))

    def extract_updates(self, body: str) -> Dict[str, str]:
        """Extract variable updates from loop body"""
        updates = {}
        # Match patterns like: x := x + 1, y := y * 2, etc.
        patterns = [
            r'(\w+)\s*:=\s*\1\s*([+\-*/])\s*(\d+)',  # x := x + 1
            r'(\w+)\s*:=\s*(\d+)\s*([+\-*/])\s*\1',  # x := 1 + x
            r'(\w+)\s*:=\s*(\w+)',  # simple assignment
        ]
        
        for pattern in patterns[:2]:
            for match in re.finditer(pattern, body):
                var = match.group(1)
                if pattern == patterns[0]:
                    op, val = match.group(2), match.group(3)
                    updates[var] = f"{op}{val}"
                else:
                    val, op = match.group(2), match.group(3)
                    updates[var] = f"{op}{val}"
        
        return updates

    def check_conditionals(self, body: str) -> bool:
        """Check if loop body contains conditionals"""
        return 'if' in body or 'else' in body

    def parse_while_loop(self) -> LoopInfo:
        """Parse a while loop"""
        self.advance()  # skip 'while'
        
        # Parse condition
        condition = self.parse_condition()
        loop_variables = self.extract_variables(condition)
        
        # Parse invariants
        invariants = []
        while self.current_token().type == TokenType.INVARIANT:
            self.advance()
            inv = self.parse_condition()
            invariants.append(inv)
        
        # Skip decreases if present
        if self.current_token().type == TokenType.DECREASES:
            self.advance()
            self.parse_condition()
        
        # Parse body
        self.expect(TokenType.LBRACE)
        body_start = self.tokens[self.pos - 1].pos
        
        brace_count = 1
        while brace_count > 0 and self.current_token().type != TokenType.EOF:
            if self.current_token().type == TokenType.LBRACE:
                brace_count += 1
            elif self.current_token().type == TokenType.RBRACE:
                brace_count -= 1
            self.advance()
        
        body_end = self.tokens[self.pos - 1].pos
        body = self.source[body_start:body_end]
        
        # Extract additional info from body
        body_vars = self.extract_variables(body)
        all_vars = list(set(loop_variables + body_vars))
        updates = self.extract_updates(body)
        has_conditionals = self.check_conditionals(body)
        
        return LoopInfo(
            variables=all_vars,
            loop_type="while",
            condition=condition,
            body=body,
            invariants=invariants,
            has_conditionals=has_conditionals,
            body_updates=updates
        )

    def parse_method_signature(self) -> tuple:
        """Parse method name, parameters, and return variables"""
        name = self.current_token().value
        self.advance()
        
        # Parse parameters
        params = []
        self.expect(TokenType.LPAREN)
        while self.current_token().type != TokenType.RPAREN:
            if self.current_token().type == TokenType.IDENTIFIER:
                param_name = self.current_token().value
                self.advance()
                if self.current_token().type == TokenType.COLON:
                    self.advance()
                    param_type = self.current_token().value
                    self.advance()
                    params.append((param_name, param_type))
            elif self.current_token().type == TokenType.COMMA:
                self.advance()
            else:
                self.advance()
        self.expect(TokenType.RPAREN)
        
        # Parse returns
        returns = []
        if self.current_token().type == TokenType.RETURNS:
            self.advance()
            self.expect(TokenType.LPAREN)
            while self.current_token().type != TokenType.RPAREN:
                if self.current_token().type == TokenType.IDENTIFIER:
                    ret_name = self.current_token().value
                    self.advance()
                    if self.current_token().type == TokenType.COLON:
                        self.advance()
                        ret_type = self.current_token().value
                        self.advance()
                        returns.append((ret_name, ret_type))
                elif self.current_token().type == TokenType.COMMA:
                    self.advance()
                else:
                    self.advance()
            self.expect(TokenType.RPAREN)
        
        return name, params, returns

    def parse_preconditions(self) -> List[str]:
        """Parse requires clauses"""
        preconditions = []
        while self.current_token().type == TokenType.REQUIRES:
            self.advance()
            cond = self.parse_condition()
            preconditions.append(cond)
        return preconditions

    def parse_postconditions(self) -> List[str]:
        """Parse ensures clauses"""
        postconditions = []
        while self.current_token().type == TokenType.ENSURES:
            self.advance()
            cond = self.parse_condition()
            postconditions.append(cond)
        return postconditions

    def parse_method_body(self) -> tuple:
        """Parse method body"""
        loops = []
        local_vars = []
        
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
                loops.append(self.parse_while_loop())
            elif token.type == TokenType.VAR:
                self.advance()
                if self.current_token().type == TokenType.IDENTIFIER:
                    local_vars.append(self.current_token().value)
                self.advance()
            else:
                self.advance()
        
        return loops, local_vars

    def parse_method(self) -> MethodInfo:
        """Parse a complete method"""
        self.expect(TokenType.METHOD)
        name, params, returns = self.parse_method_signature()
        preconditions = self.parse_preconditions()
        postconditions = self.parse_postconditions()
        loops, local_vars = self.parse_method_body()
        
        return MethodInfo(
            name=name,
            parameters=params,
            return_vars=returns,
            preconditions=preconditions,
            postconditions=postconditions,
            loops=loops,
            local_variables=local_vars
        )

    def parse(self) -> List[MethodInfo]:
        """Parse entire program"""
        methods = []
        while self.current_token().type != TokenType.EOF:
            if self.current_token().type == TokenType.METHOD:
                methods.append(self.parse_method())
            else:
                self.advance()
        return methods


class DafnyExtractor:
    """Main extraction interface"""

    def parse_file(self, filename: str) -> Dict[str, Any]:
        try:
            with open(filename, 'r') as f:
                source = f.read()
            return self.parse_source(source)
        except FileNotFoundError:
            return {"error": f"File not found: {filename}"}
        except Exception as e:
            return {"error": f"Parse error: {str(e)}"}

    def parse_source(self, source: str) -> Dict[str, Any]:
        try:
            lexer = DafnyLexer(source)
            tokens = lexer.tokenize()
            parser = DafnyParser(tokens, source)
            methods = parser.parse()
            
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
                    "return_vars": method.return_vars,
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
                        "invariants": loop.invariants,
                        "has_conditionals": loop.has_conditionals,
                        "body_updates": loop.body_updates
                    }
                    method_data["loops"].append(loop_data)
                    result["loops"].append(loop_data)
                
                result["methods"].append(method_data)
                result["preconditions"].extend(method.preconditions)
                result["postconditions"].extend(method.postconditions)
            
            return result
        except Exception as e:
            return {"error": f"Parse error: {str(e)}"}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dafny_parser.py <file.dfy>")
        sys.exit(1)
    
    extractor = DafnyExtractor()
    result = extractor.parse_file(sys.argv[1])
    print(json.dumps(result, indent=2))
