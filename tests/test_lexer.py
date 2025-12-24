"""
Tests for the Gojju lexer.
"""

import pytest
from gojju.lexer import Lexer, Token, TokenType, LexerError


class TestLexerBasics:
    """Test basic lexer functionality."""
    
    def test_empty_input(self):
        lexer = Lexer("")
        tokens = lexer.tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF
    
    def test_numbers(self):
        lexer = Lexer("42 3.14 1_000_000")
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 42
        
        assert tokens[1].type == TokenType.NUMBER
        assert tokens[1].value == 3.14
        
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == 1000000
    
    def test_strings(self):
        lexer = Lexer('"hello" \'world\'')
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"
        
        assert tokens[1].type == TokenType.STRING
        assert tokens[1].value == "world"
    
    def test_string_escapes(self):
        lexer = Lexer(r'"hello\nworld\ttab"')
        tokens = lexer.tokenize()
        
        assert tokens[0].value == "hello\nworld\ttab"
    
    def test_identifiers(self):
        lexer = Lexer("foo bar_baz camelCase")
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "foo"
        
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "bar_baz"
    
    def test_keywords(self):
        lexer = Lexer("let mut fn if else while for in")
        tokens = lexer.tokenize()
        
        expected = [
            TokenType.LET, TokenType.MUT, TokenType.FN,
            TokenType.IF, TokenType.ELSE, TokenType.WHILE,
            TokenType.FOR, TokenType.IN
        ]
        
        for i, expected_type in enumerate(expected):
            assert tokens[i].type == expected_type
    
    def test_operators(self):
        lexer = Lexer("+ - * / % == != < > <= >= |>")
        tokens = lexer.tokenize()
        
        expected = [
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR,
            TokenType.SLASH, TokenType.PERCENT, TokenType.EQ,
            TokenType.NE, TokenType.LT, TokenType.GT,
            TokenType.LE, TokenType.GE, TokenType.PIPE
        ]
        
        for i, expected_type in enumerate(expected):
            assert tokens[i].type == expected_type
    
    def test_symbols(self):
        lexer = Lexer(":success :error :pending")
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.SYMBOL
        assert tokens[0].value == "success"
        
        assert tokens[1].type == TokenType.SYMBOL
        assert tokens[1].value == "error"
    
    def test_arrows(self):
        lexer = Lexer("-> => |>")
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.ARROW
        assert tokens[1].type == TokenType.FAT_ARROW
        assert tokens[2].type == TokenType.PIPE
    
    def test_comments(self):
        lexer = Lexer("42 # this is a comment\n43")
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 42
        
        # Newline token
        assert tokens[1].type == TokenType.NEWLINE
        
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == 43
    
    def test_template_string(self):
        lexer = Lexer('`hello world`')
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello world"


class TestLexerErrors:
    """Test lexer error handling."""
    
    def test_unterminated_string(self):
        lexer = Lexer('"hello')
        with pytest.raises(LexerError) as exc_info:
            lexer.tokenize()
        assert "Unterminated string" in str(exc_info.value)
    
    def test_unterminated_template(self):
        lexer = Lexer('`hello')
        with pytest.raises(LexerError) as exc_info:
            lexer.tokenize()
        assert "Unterminated template" in str(exc_info.value)


class TestLexerComplex:
    """Test complex lexer scenarios."""
    
    def test_function_definition(self):
        code = """
fn greet(name)
  "Hello, " + name
end
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        
        # Find key tokens
        token_types = [t.type for t in tokens if t.type != TokenType.NEWLINE]
        
        assert TokenType.FN in token_types
        assert TokenType.IDENTIFIER in token_types
        assert TokenType.LPAREN in token_types
        assert TokenType.RPAREN in token_types
        assert TokenType.STRING in token_types
        assert TokenType.PLUS in token_types
        assert TokenType.END in token_types
    
    def test_pipe_expression(self):
        code = "[1, 2, 3] |> map(double) |> sum"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        
        pipe_tokens = [t for t in tokens if t.type == TokenType.PIPE]
        assert len(pipe_tokens) == 2
    
    def test_lambda_syntax(self):
        code = r"\x -> x * 2"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.LAMBDA
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[2].type == TokenType.ARROW
