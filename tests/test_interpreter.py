"""
Tests for the Gojju interpreter.
"""

import pytest
from gojju import run
from gojju.interpreter import GojjuRuntimeError


class TestInterpreterLiterals:
    """Test interpreting literals."""
    
    def test_numbers(self):
        assert run("42") == 42
        assert run("3.14") == 3.14
    
    def test_strings(self):
        assert run('"hello"') == "hello"
    
    def test_booleans(self):
        assert run("true") == True
        assert run("false") == False
    
    def test_nil(self):
        assert run("nil") is None
    
    def test_lists(self):
        assert run("[1, 2, 3]") == [1, 2, 3]
        assert run("[]") == []
    
    def test_dicts(self):
        assert run('{"a": 1}') == {"a": 1}


class TestInterpreterExpressions:
    """Test interpreting expressions."""
    
    def test_arithmetic(self):
        assert run("1 + 2") == 3
        assert run("10 - 3") == 7
        assert run("4 * 5") == 20
        assert run("20 / 4") == 5.0
        assert run("7 % 3") == 1
        assert run("2 ** 3") == 8
    
    def test_comparisons(self):
        assert run("1 < 2") == True
        assert run("2 > 1") == True
        assert run("1 <= 1") == True
        assert run("1 >= 1") == True
        assert run("1 == 1") == True
        assert run("1 != 2") == True
    
    def test_logical(self):
        assert run("true and true") == True
        assert run("true and false") == False
        assert run("true or false") == True
        assert run("false or false") == False
        assert run("not true") == False
        assert run("not false") == True
    
    def test_string_concatenation(self):
        assert run('"hello" + " " + "world"') == "hello world"
    
    def test_list_concatenation(self):
        assert run("[1, 2] + [3, 4]") == [1, 2, 3, 4]


class TestInterpreterVariables:
    """Test variable handling."""
    
    def test_let_binding(self):
        assert run("let x = 42\nx") == 42
    
    def test_mutable_variable(self):
        assert run("mut x = 0\nx = 42\nx") == 42
    
    def test_variable_in_expression(self):
        assert run("let x = 10\nlet y = 20\nx + y") == 30


class TestInterpreterFunctions:
    """Test function definitions and calls."""
    
    def test_simple_function(self):
        code = """
fn double(n)
  n * 2
end
double(21)
"""
        assert run(code) == 42
    
    def test_multi_param_function(self):
        code = """
fn add(a, b)
  a + b
end
add(10, 32)
"""
        assert run(code) == 42
    
    def test_recursive_function(self):
        code = """
fn factorial(n)
  if n <= 1
    1
  else
    n * factorial(n - 1)
  end
end
factorial(5)
"""
        assert run(code) == 120
    
    def test_closure(self):
        code = """
fn make_adder(x)
  (y) => x + y
end
let add5 = make_adder(5)
add5(10)
"""
        assert run(code) == 15


class TestInterpreterLambdas:
    """Test lambda expressions."""
    
    def test_arrow_function(self):
        code = """
let double = (x) => x * 2
double(21)
"""
        assert run(code) == 42
    
    def test_haskell_lambda(self):
        code = r"""
let inc = \x -> x + 1
inc(41)
"""
        assert run(code) == 42


class TestInterpreterControlFlow:
    """Test control flow."""
    
    def test_if_true(self):
        code = """
if true
  42
else
  0
end
"""
        assert run(code) == 42
    
    def test_if_false(self):
        code = """
if false
  0
else
  42
end
"""
        assert run(code) == 42
    
    def test_while_loop(self):
        code = """
mut sum = 0
mut i = 1
while i <= 5
  sum = sum + i
  i = i + 1
end
sum
"""
        assert run(code) == 15
    
    def test_for_loop(self):
        code = """
mut sum = 0
for i in [1, 2, 3, 4, 5]
  sum = sum + i
end
sum
"""
        assert run(code) == 15


class TestInterpreterPatternMatching:
    """Test pattern matching."""
    
    def test_literal_matching(self):
        code = """
let x = 1
match x
  0 -> "zero"
  1 -> "one"
  _ -> "other"
end
"""
        assert run(code) == "one"
    
    def test_wildcard_matching(self):
        code = """
match 42
  0 -> "zero"
  _ -> "other"
end
"""
        assert run(code) == "other"
    
    def test_guard_matching(self):
        code = """
let x = -5
match x
  x if x < 0 -> "negative"
  _ -> "non-negative"
end
"""
        assert run(code) == "negative"


class TestInterpreterBuiltins:
    """Test built-in functions."""
    
    def test_len(self):
        assert run('len([1, 2, 3])') == 3
        assert run('len("hello")') == 5
    
    def test_range(self):
        assert run('range(5)') == [0, 1, 2, 3, 4]
        assert run('range(1, 4)') == [1, 2, 3]
    
    def test_head_tail(self):
        assert run('head([1, 2, 3])') == 1
        assert run('tail([1, 2, 3])') == [2, 3]
    
    def test_map(self):
        code = r'map(\x -> x * 2, [1, 2, 3])'
        assert run(code) == [2, 4, 6]
    
    def test_filter(self):
        code = r'filter(\x -> x % 2 == 0, [1, 2, 3, 4])'
        assert run(code) == [2, 4]
    
    def test_reduce(self):
        code = r'reduce(\acc x -> acc + x, [1, 2, 3, 4, 5], 0)'
        assert run(code) == 15
    
    def test_sum(self):
        assert run('sum([1, 2, 3, 4, 5])') == 15


class TestInterpreterPipe:
    """Test pipe operator."""
    
    def test_simple_pipe(self):
        code = r'5 |> \x -> x * 2'
        assert run(code) == 10
    
    def test_chained_pipe(self):
        code = r"""
let double = \x -> x * 2
let inc = \x -> x + 1
5 |> inc |> double
"""
        assert run(code) == 12


class TestInterpreterListComprehension:
    """Test list comprehensions."""
    
    def test_simple_comprehension(self):
        assert run('[x * 2 for x in range(5)]') == [0, 2, 4, 6, 8]
    
    def test_filtered_comprehension(self):
        assert run('[x for x in range(10) if x % 2 == 0]') == [0, 2, 4, 6, 8]


class TestInterpreterMethods:
    """Test method calls on objects."""
    
    def test_list_map(self):
        code = r'[1, 2, 3].map(\x -> x * 2)'
        assert run(code) == [2, 4, 6]
    
    def test_list_filter(self):
        code = r'[1, 2, 3, 4].filter(\x -> x % 2 == 0)'
        assert run(code) == [2, 4]
    
    def test_string_upper(self):
        assert run('"hello".upper()') == "HELLO"
    
    def test_string_split(self):
        assert run('"a,b,c".split(",")') == ["a", "b", "c"]


class TestInterpreterPostfixConditionals:
    """Test postfix conditionals."""
    
    def test_postfix_if_true(self):
        assert run('42 if true') == 42
    
    def test_postfix_if_false(self):
        assert run('42 if false') is None
    
    def test_postfix_unless_true(self):
        assert run('42 unless true') is None
    
    def test_postfix_unless_false(self):
        assert run('42 unless false') == 42
