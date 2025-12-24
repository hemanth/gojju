"""
Tests for the Gojju parser.
"""

import pytest
from gojju.lexer import Lexer
from gojju.parser import Parser, ParseError
from gojju.ast import *


def parse(code: str) -> Program:
    """Helper to parse code."""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()


class TestParserExpressions:
    """Test parsing expressions."""
    
    def test_number_literal(self):
        ast = parse("42")
        assert len(ast.statements) == 1
        assert isinstance(ast.statements[0], NumberLiteral)
        assert ast.statements[0].value == 42
    
    def test_string_literal(self):
        ast = parse('"hello"')
        assert isinstance(ast.statements[0], StringLiteral)
        assert ast.statements[0].value == "hello"
    
    def test_boolean_literals(self):
        ast = parse("true")
        assert isinstance(ast.statements[0], BooleanLiteral)
        assert ast.statements[0].value == True
        
        ast = parse("false")
        assert isinstance(ast.statements[0], BooleanLiteral)
        assert ast.statements[0].value == False
    
    def test_nil_literal(self):
        ast = parse("nil")
        assert isinstance(ast.statements[0], NilLiteral)
    
    def test_symbol_literal(self):
        ast = parse(":success")
        assert isinstance(ast.statements[0], SymbolLiteral)
        assert ast.statements[0].name == "success"
    
    def test_list_literal(self):
        ast = parse("[1, 2, 3]")
        assert isinstance(ast.statements[0], ListLiteral)
        assert len(ast.statements[0].elements) == 3
    
    def test_dict_literal(self):
        ast = parse('{"a": 1, "b": 2}')
        assert isinstance(ast.statements[0], DictLiteral)
        assert len(ast.statements[0].pairs) == 2
    
    def test_binary_operations(self):
        ast = parse("1 + 2")
        assert isinstance(ast.statements[0], BinaryOp)
        assert ast.statements[0].operator == '+'
        
        ast = parse("3 * 4")
        assert isinstance(ast.statements[0], BinaryOp)
        assert ast.statements[0].operator == '*'
    
    def test_operator_precedence(self):
        ast = parse("1 + 2 * 3")
        expr = ast.statements[0]
        assert isinstance(expr, BinaryOp)
        assert expr.operator == '+'
        assert isinstance(expr.right, BinaryOp)
        assert expr.right.operator == '*'
    
    def test_unary_operations(self):
        ast = parse("-5")
        assert isinstance(ast.statements[0], UnaryOp)
        assert ast.statements[0].operator == '-'
        
        ast = parse("not true")
        assert isinstance(ast.statements[0], UnaryOp)
        assert ast.statements[0].operator == 'not'
    
    def test_function_call(self):
        ast = parse("print(42)")
        assert isinstance(ast.statements[0], Call)
        assert isinstance(ast.statements[0].callee, Identifier)
    
    def test_indexing(self):
        ast = parse("list[0]")
        assert isinstance(ast.statements[0], Index)
    
    def test_slicing(self):
        ast = parse("list[1:3]")
        assert isinstance(ast.statements[0], Slice)
    
    def test_property_access(self):
        ast = parse("obj.property")
        assert isinstance(ast.statements[0], PropertyAccess)
    
    def test_pipe_expression(self):
        ast = parse("x |> double")
        assert isinstance(ast.statements[0], PipeExpr)


class TestParserLambdas:
    """Test parsing lambda expressions."""
    
    def test_arrow_function(self):
        ast = parse("(x) => x * 2")
        assert isinstance(ast.statements[0], Lambda)
        assert ast.statements[0].params == ['x']
    
    def test_multi_param_arrow(self):
        ast = parse("(a, b) => a + b")
        assert isinstance(ast.statements[0], Lambda)
        assert ast.statements[0].params == ['a', 'b']
    
    def test_haskell_lambda(self):
        ast = parse(r"\x -> x + 1")
        assert isinstance(ast.statements[0], Lambda)
        assert ast.statements[0].params == ['x']
    
    def test_multi_param_haskell_lambda(self):
        ast = parse(r"\x y -> x + y")
        assert isinstance(ast.statements[0], Lambda)
        assert ast.statements[0].params == ['x', 'y']


class TestParserStatements:
    """Test parsing statements."""
    
    def test_let_statement(self):
        ast = parse("let x = 42")
        assert isinstance(ast.statements[0], LetStatement)
        assert ast.statements[0].name == 'x'
        assert ast.statements[0].mutable == False
    
    def test_mut_statement(self):
        ast = parse("mut x = 42")
        assert isinstance(ast.statements[0], LetStatement)
        assert ast.statements[0].mutable == True
    
    def test_assignment(self):
        ast = parse("let x = 0\nx = 42")
        assert isinstance(ast.statements[1], Assignment)
    
    def test_if_expression(self):
        ast = parse("""
if x > 0
  "positive"
end
""")
        assert isinstance(ast.statements[0], IfExpr)
    
    def test_if_else(self):
        ast = parse("""
if x > 0
  "positive"
else
  "non-positive"
end
""")
        assert isinstance(ast.statements[0], IfExpr)
        assert ast.statements[0].else_branch is not None
    
    def test_unless(self):
        ast = parse("""
unless error
  continue
end
""")
        assert isinstance(ast.statements[0], IfExpr)
        assert ast.statements[0].is_unless == True
    
    def test_while_loop(self):
        ast = parse("""
while x < 10
  x = x + 1
end
""")
        assert isinstance(ast.statements[0], WhileLoop)
    
    def test_until_loop(self):
        ast = parse("""
until x == 10
  x = x + 1
end
""")
        assert isinstance(ast.statements[0], WhileLoop)
        assert ast.statements[0].is_until == True
    
    def test_for_loop(self):
        ast = parse("""
for x in range(10)
  print(x)
end
""")
        assert isinstance(ast.statements[0], ForLoop)
        assert ast.statements[0].var == 'x'
    
    def test_return_statement(self):
        ast = parse("return 42")
        assert isinstance(ast.statements[0], ReturnStatement)
    
    def test_postfix_if(self):
        ast = parse('print("yes") if happy')
        assert isinstance(ast.statements[0], PostfixIf)
        assert ast.statements[0].is_unless == False
    
    def test_postfix_unless(self):
        ast = parse('print("yes") unless sad')
        assert isinstance(ast.statements[0], PostfixIf)
        assert ast.statements[0].is_unless == True


class TestParserFunctions:
    """Test parsing function definitions."""
    
    def test_simple_function(self):
        ast = parse("""
fn double(n)
  n * 2
end
""")
        assert isinstance(ast.statements[0], FunctionDef)
        assert ast.statements[0].name == 'double'
        assert ast.statements[0].params == ['n']
    
    def test_multi_param_function(self):
        ast = parse("""
fn add(a, b)
  a + b
end
""")
        assert isinstance(ast.statements[0], FunctionDef)
        assert ast.statements[0].params == ['a', 'b']
    
    def test_no_param_function(self):
        ast = parse("""
fn greet()
  "Hello!"
end
""")
        assert isinstance(ast.statements[0], FunctionDef)
        assert ast.statements[0].params == []


class TestParserPatternMatching:
    """Test parsing pattern matching."""
    
    def test_simple_match(self):
        ast = parse("""
match x
  0 -> "zero"
  1 -> "one"
  _ -> "other"
end
""")
        assert isinstance(ast.statements[0], MatchExpr)
        assert len(ast.statements[0].cases) == 3
    
    def test_match_with_guard(self):
        ast = parse("""
match n
  n if n < 0 -> "negative"
  _ -> "non-negative"
end
""")
        assert isinstance(ast.statements[0], MatchExpr)
        assert ast.statements[0].cases[0].guard is not None


class TestParserListComprehension:
    """Test parsing list comprehensions."""
    
    def test_simple_comprehension(self):
        ast = parse("[x * 2 for x in range(10)]")
        assert isinstance(ast.statements[0], ListComprehension)
    
    def test_comprehension_with_condition(self):
        ast = parse("[x for x in range(10) if x % 2 == 0]")
        assert isinstance(ast.statements[0], ListComprehension)
        assert ast.statements[0].condition is not None
