"""
Gojju - A programming language combining the best of Python, Ruby, Haskell, Perl, and JavaScript.

Usage:
    from gojju import run, run_file, Interpreter
    
    # Run code directly
    result = run('let x = 5; x * 2')
    
    # Run a file
    result = run_file('script.gj')
    
    # Use interpreter directly
    interp = Interpreter()
    interp.execute('let x = 10')
    interp.execute('print x')
"""

__version__ = "0.1.4"
__author__ = "Hemanth HM"

from .lexer import Lexer, Token, TokenType
from .parser import Parser
from .ast import *
from .interpreter import Interpreter
from .builtins import BUILTINS
from .fp import (
    Maybe, Either, Lazy,
    compose, pipe, curry2, curry3, flip, identity, partial, memoize,
    foldl, foldr, scanl, scanr, unfold, iterate,
    partition, group_by, transpose,
    FP_BUILTINS
)

def run(source: str):
    """Execute Gojju source code and return the result."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    interpreter = Interpreter()
    return interpreter.execute(ast)

def run_file(filepath: str):
    """Execute a Gojju source file and return the result."""
    with open(filepath, 'r') as f:
        source = f.read()
    return run(source)

__all__ = [
    'run',
    'run_file',
    'Lexer',
    'Token',
    'TokenType',
    'Parser',
    'Interpreter',
    'BUILTINS',
    '__version__',
]
