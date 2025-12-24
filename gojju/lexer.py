"""
Lexer (Tokenizer) for the Gojju programming language.
Converts source code into a stream of tokens.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, List, Optional
import re


class TokenType(Enum):
    """Token types for Gojju."""
    # Literals
    NUMBER = auto()
    STRING = auto()
    SYMBOL = auto()
    REGEX = auto()
    TRUE = auto()
    FALSE = auto()
    NIL = auto()
    
    # Identifiers and Keywords
    IDENTIFIER = auto()
    
    # Keywords
    LET = auto()
    MUT = auto()
    FN = auto()
    IF = auto()
    ELSE = auto()
    UNLESS = auto()
    WHILE = auto()
    UNTIL = auto()
    FOR = auto()
    IN = auto()
    DO = auto()
    END = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    MATCH = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Operators
    PLUS = auto()          # +
    MINUS = auto()         # -
    STAR = auto()          # *
    SLASH = auto()         # /
    PERCENT = auto()       # %
    POWER = auto()         # **
    
    EQ = auto()            # ==
    NE = auto()            # !=
    LT = auto()            # <
    GT = auto()            # >
    LE = auto()            # <=
    GE = auto()            # >=
    
    ASSIGN = auto()        # =
    PLUS_ASSIGN = auto()   # +=
    MINUS_ASSIGN = auto()  # -=
    STAR_ASSIGN = auto()   # *=
    SLASH_ASSIGN = auto()  # /=
    
    PIPE = auto()          # |>
    ARROW = auto()         # ->
    FAT_ARROW = auto()     # =>
    LAMBDA = auto()        # \
    
    DOT = auto()           # .
    OPTIONAL_DOT = auto()  # ?.
    RANGE = auto()         # ..
    SPREAD = auto()        # ...
    
    # Delimiters
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    LBRACKET = auto()      # [
    RBRACKET = auto()      # ]
    LBRACE = auto()        # {
    RBRACE = auto()        # }
    COMMA = auto()         # ,
    COLON = auto()         # :
    SEMICOLON = auto()     # ;
    PIPE_CHAR = auto()     # | (for block params)
    HASH = auto()          # #
    NEWLINE = auto()
    
    # Special
    EOF = auto()
    

@dataclass
class Token:
    """A lexical token."""
    type: TokenType
    value: Any
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


# Keyword mapping
KEYWORDS = {
    'let': TokenType.LET,
    'mut': TokenType.MUT,
    'fn': TokenType.FN,
    'if': TokenType.IF,
    'else': TokenType.ELSE,
    'unless': TokenType.UNLESS,
    'while': TokenType.WHILE,
    'until': TokenType.UNTIL,
    'for': TokenType.FOR,
    'in': TokenType.IN,
    'do': TokenType.DO,
    'end': TokenType.END,
    'return': TokenType.RETURN,
    'break': TokenType.BREAK,
    'continue': TokenType.CONTINUE,
    'match': TokenType.MATCH,
    'and': TokenType.AND,
    'or': TokenType.OR,
    'not': TokenType.NOT,
    'true': TokenType.TRUE,
    'false': TokenType.FALSE,
    'nil': TokenType.NIL,
}


class LexerError(Exception):
    """Lexer error with position information."""
    def __init__(self, message: str, line: int, column: int):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Line {line}, Column {column}: {message}")


class Lexer:
    """Tokenizer for Gojju source code."""
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
    
    @property
    def current(self) -> Optional[str]:
        """Current character."""
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]
    
    def peek(self, offset: int = 1) -> Optional[str]:
        """Peek ahead by offset characters."""
        pos = self.pos + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self) -> Optional[str]:
        """Advance to next character and return current."""
        char = self.current
        if char is not None:
            self.pos += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
        return char
    
    def skip_whitespace(self):
        """Skip whitespace (but not newlines in significant places)."""
        while self.current in ' \t\r':
            self.advance()
    
    def skip_comment(self):
        """Skip a comment (# to end of line)."""
        while self.current is not None and self.current != '\n':
            self.advance()
    
    def add_token(self, type: TokenType, value: Any = None):
        """Add a token to the list."""
        self.tokens.append(Token(type, value, self.line, self.column))
    
    def read_string(self, quote: str) -> str:
        """Read a string literal."""
        start_line = self.line
        start_col = self.column
        self.advance()  # Skip opening quote
        
        result = []
        while self.current is not None:
            if self.current == quote:
                self.advance()  # Skip closing quote
                return ''.join(result)
            elif self.current == '\\':
                self.advance()
                escape_char = self.advance()
                if escape_char == 'n':
                    result.append('\n')
                elif escape_char == 't':
                    result.append('\t')
                elif escape_char == 'r':
                    result.append('\r')
                elif escape_char == '\\':
                    result.append('\\')
                elif escape_char == quote:
                    result.append(quote)
                elif escape_char == '#':
                    result.append('#')
                elif escape_char == '$':
                    result.append('$')
                else:
                    result.append('\\')
                    if escape_char:
                        result.append(escape_char)
            else:
                result.append(self.advance())
        
        raise LexerError("Unterminated string", start_line, start_col)
    
    def read_template_string(self) -> str:
        """Read a template string (backtick delimited)."""
        start_line = self.line
        start_col = self.column
        self.advance()  # Skip opening backtick
        
        result = []
        while self.current is not None:
            if self.current == '`':
                self.advance()  # Skip closing backtick
                return ''.join(result)
            elif self.current == '\\':
                self.advance()
                escape_char = self.advance()
                if escape_char == 'n':
                    result.append('\n')
                elif escape_char == 't':
                    result.append('\t')
                elif escape_char == '`':
                    result.append('`')
                elif escape_char == '$':
                    result.append('$')
                else:
                    result.append('\\')
                    if escape_char:
                        result.append(escape_char)
            else:
                result.append(self.advance())
        
        raise LexerError("Unterminated template string", start_line, start_col)
    
    def read_number(self) -> Token:
        """Read a numeric literal."""
        start_col = self.column
        result = []
        has_dot = False
        
        while self.current is not None:
            if self.current.isdigit():
                result.append(self.advance())
            elif self.current == '.' and not has_dot and self.peek() and self.peek().isdigit():
                has_dot = True
                result.append(self.advance())
            elif self.current == '_':  # Allow underscores in numbers
                self.advance()
            else:
                break
        
        value_str = ''.join(result)
        value = float(value_str) if has_dot else int(value_str)
        return Token(TokenType.NUMBER, value, self.line, start_col)
    
    def read_identifier(self) -> Token:
        """Read an identifier or keyword."""
        start_col = self.column
        result = []
        
        while self.current is not None and (self.current.isalnum() or self.current in '_?!'):
            result.append(self.advance())
        
        name = ''.join(result)
        token_type = KEYWORDS.get(name, TokenType.IDENTIFIER)
        return Token(token_type, name, self.line, start_col)
    
    def read_symbol(self) -> Token:
        """Read a symbol literal (:name)."""
        start_col = self.column
        self.advance()  # Skip the colon
        
        result = []
        while self.current is not None and (self.current.isalnum() or self.current == '_'):
            result.append(self.advance())
        
        if not result:
            raise LexerError("Expected symbol name after ':'", self.line, start_col)
        
        return Token(TokenType.SYMBOL, ''.join(result), self.line, start_col)
    
    def read_regex(self) -> Token:
        """Read a regex literal (/pattern/flags)."""
        start_line = self.line
        start_col = self.column
        self.advance()  # Skip opening /
        
        pattern = []
        while self.current is not None and self.current != '/':
            if self.current == '\\' and self.peek() == '/':
                pattern.append(self.advance())
                pattern.append(self.advance())
            elif self.current == '\n':
                raise LexerError("Unterminated regex", start_line, start_col)
            else:
                pattern.append(self.advance())
        
        if self.current != '/':
            raise LexerError("Unterminated regex", start_line, start_col)
        
        self.advance()  # Skip closing /
        
        # Read flags
        flags = []
        while self.current is not None and self.current in 'gimsux':
            flags.append(self.advance())
        
        return Token(TokenType.REGEX, (''.join(pattern), ''.join(flags)), self.line, start_col)
    
    def tokenize(self) -> List[Token]:
        """Tokenize the source code."""
        while self.current is not None:
            # Skip whitespace
            if self.current in ' \t\r':
                self.skip_whitespace()
                continue
            
            # Comments
            if self.current == '#':
                self.skip_comment()
                continue
            
            # Newlines (significant for some constructs)
            if self.current == '\n':
                self.add_token(TokenType.NEWLINE, '\n')
                self.advance()
                continue
            
            start_col = self.column
            
            # Numbers
            if self.current.isdigit():
                self.tokens.append(self.read_number())
                continue
            
            # Strings
            if self.current in '"\'':
                quote = self.current
                value = self.read_string(quote)
                self.add_token(TokenType.STRING, value)
                continue
            
            # Template strings
            if self.current == '`':
                value = self.read_template_string()
                self.add_token(TokenType.STRING, value)
                continue
            
            # Identifiers and keywords
            if self.current.isalpha() or self.current == '_':
                self.tokens.append(self.read_identifier())
                continue
            
            # Symbols (:name) vs just colon
            if self.current == ':':
                if self.peek() and (self.peek().isalpha() or self.peek() == '_'):
                    self.tokens.append(self.read_symbol())
                else:
                    self.advance()
                    self.add_token(TokenType.COLON, ':')
                continue
            
            # Multi-character operators
            if self.current == '|':
                self.advance()
                if self.current == '>':
                    self.advance()
                    self.add_token(TokenType.PIPE, '|>')
                else:
                    self.add_token(TokenType.PIPE_CHAR, '|')
                continue
            
            if self.current == '-':
                self.advance()
                if self.current == '>':
                    self.advance()
                    self.add_token(TokenType.ARROW, '->')
                elif self.current == '=':
                    self.advance()
                    self.add_token(TokenType.MINUS_ASSIGN, '-=')
                else:
                    self.add_token(TokenType.MINUS, '-')
                continue
            
            if self.current == '=':
                self.advance()
                if self.current == '=':
                    self.advance()
                    self.add_token(TokenType.EQ, '==')
                elif self.current == '>':
                    self.advance()
                    self.add_token(TokenType.FAT_ARROW, '=>')
                else:
                    self.add_token(TokenType.ASSIGN, '=')
                continue
            
            if self.current == '+':
                self.advance()
                if self.current == '=':
                    self.advance()
                    self.add_token(TokenType.PLUS_ASSIGN, '+=')
                else:
                    self.add_token(TokenType.PLUS, '+')
                continue
            
            if self.current == '*':
                self.advance()
                if self.current == '*':
                    self.advance()
                    self.add_token(TokenType.POWER, '**')
                elif self.current == '=':
                    self.advance()
                    self.add_token(TokenType.STAR_ASSIGN, '*=')
                else:
                    self.add_token(TokenType.STAR, '*')
                continue
            
            if self.current == '/':
                # Could be division, /= or regex
                # For simplicity, treat / at start of expression as regex
                # This is a simplification - real parser would use context
                self.advance()
                if self.current == '=':
                    self.advance()
                    self.add_token(TokenType.SLASH_ASSIGN, '/=')
                else:
                    self.add_token(TokenType.SLASH, '/')
                continue
            
            if self.current == '%':
                self.advance()
                self.add_token(TokenType.PERCENT, '%')
                continue
            
            if self.current == '<':
                self.advance()
                if self.current == '=':
                    self.advance()
                    self.add_token(TokenType.LE, '<=')
                else:
                    self.add_token(TokenType.LT, '<')
                continue
            
            if self.current == '>':
                self.advance()
                if self.current == '=':
                    self.advance()
                    self.add_token(TokenType.GE, '>=')
                else:
                    self.add_token(TokenType.GT, '>')
                continue
            
            if self.current == '!':
                self.advance()
                if self.current == '=':
                    self.advance()
                    self.add_token(TokenType.NE, '!=')
                else:
                    self.add_token(TokenType.NOT, 'not')
                continue
            
            if self.current == '.':
                self.advance()
                if self.current == '.':
                    self.advance()
                    if self.current == '.':
                        self.advance()
                        self.add_token(TokenType.SPREAD, '...')
                    else:
                        self.add_token(TokenType.RANGE, '..')
                else:
                    self.add_token(TokenType.DOT, '.')
                continue
            
            if self.current == '?':
                self.advance()
                if self.current == '.':
                    self.advance()
                    self.add_token(TokenType.OPTIONAL_DOT, '?.')
                else:
                    # Just a question mark (could be used elsewhere)
                    self.add_token(TokenType.IDENTIFIER, '?')
                continue
            
            if self.current == '\\':
                self.advance()
                self.add_token(TokenType.LAMBDA, '\\')
                continue
            
            # Single-character tokens
            simple_tokens = {
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                ',': TokenType.COMMA,
                ';': TokenType.SEMICOLON,
            }
            
            if self.current in simple_tokens:
                char = self.current
                self.advance()
                self.add_token(simple_tokens[char], char)
                continue
            
            # Unknown character
            raise LexerError(f"Unexpected character: {self.current!r}", self.line, self.column)
        
        # Add EOF token
        self.add_token(TokenType.EOF, None)
        return self.tokens
